//! The race: SGLang's own chat formatter for a model (its hand-written renderer or its Jinja
//! path, then the tokenizer) against the Affordance formatter, on conversations that grow by a
//! turn a request, as an agent's do. The prompt ids must be equal at every step.
//!
//! Needs an affordance checkout with its reference models:
//!
//!   AFFORDANCE_ROOT=/path/to/affordance AFFORDANCE_MODEL=zai-org__GLM-5.3 \
//!     cargo test -p sglang-processor --release --test affordance_race -- --ignored --nocapture

#![cfg(all(feature = "render", feature = "tokenizer"))]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde_json::{Value, json};
use sglang_processor::dynamo_tokenizers::Tokenizer;
use sglang_processor::{ChatFormatter, ChatFormatterOptions, load_tokenizer, select_chat_formatter};

/// Where SGLang's time goes when it renders text and then tokenizes it.
static RENDER_NS: AtomicU64 = AtomicU64::new(0);
static TOKENIZE_NS: AtomicU64 = AtomicU64::new(0);

struct Race {
    model: String,
    theirs: ChatFormatter,
    tokenizer: Tokenizer,
    ours: ChatFormatter,
}

impl Race {
    fn load() -> Self {
        let root = PathBuf::from(std::env::var("AFFORDANCE_ROOT").unwrap());
        let model = std::env::var("AFFORDANCE_MODEL").unwrap();
        let dir = root.join("ref/models").join(&model);
        let (theirs, error) = select_chat_formatter(&ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
        let index: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("views/index.json")).unwrap()).unwrap();
        let mut options = affordance::chat_render::Options::new(
            root.join("views").join(index[&model]["view"].as_str().unwrap()),
            index[&model]["entry"].as_str().unwrap(),
            root.join(format!("ref/tokenizers/{model}.json")),
        );
        options.special = serde_json::from_str(
            &std::fs::read_to_string(root.join(format!("ref/tokenizers/{model}.special.json"))).unwrap(),
        )
        .unwrap();
        Race {
            theirs: theirs.clone().unwrap_or_else(|| panic!("{error:?}")),
            tokenizer: load_tokenizer(Some(dir.join("tokenizer.json").to_str().unwrap()), None, true).unwrap(),
            ours: ChatFormatter::Affordance {
                renderer: Arc::new(
                    affordance::chat_render::ChatRenderer::new(options, sglang_processor::affordance_message).unwrap(),
                ),
                own: Box::new(theirs.clone().unwrap()),
            },
            model,
        }
    }

    fn kind(&self) -> &'static str {
        match &self.theirs {
            ChatFormatter::DeepSeekV4(_) | ChatFormatter::DeepSeekV41(_) => "its DeepSeek renderer (deepseek-recipe)",
            ChatFormatter::Glm53(_) => "its GLM-5.3 renderer over Jinja",
            ChatFormatter::Qwen38 { .. } => "its Qwen3.8 renderer over Jinja",
            ChatFormatter::HuggingFace { .. } => "Jinja",
            _ => "another formatter",
        }
    }

    /// The prompt ids SGLang encodes, as its server does: the formatter's own ids when it
    /// encodes requests, else the rendered prompt through the tokenizer.
    fn theirs(&self, request: &Value) -> Vec<u32> {
        let defaults = HashMap::new();
        if let Some(ids) = self.theirs.encode_request(request.clone(), &defaults) {
            return ids.unwrap();
        }
        let t = Instant::now();
        let (prompt, _) = self.theirs.render_request(request.clone(), &defaults).unwrap();
        RENDER_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t = Instant::now();
        let ids = self.tokenizer.encode(prompt.as_str()).unwrap().token_ids().to_vec();
        TOKENIZE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        ids
    }

    fn ours(&self, request: &Value) -> Vec<u32> {
        self.ours.encode_request(request.clone(), &HashMap::new()).unwrap().unwrap()
    }
}

fn request(messages: &[Value]) -> Value {
    let mut request = json!({"model": "m", "messages": messages});
    // SGLang's DeepSeek renderer reasons only when asked; the official template reasons unless
    // told not to. The race states the mode, so both render the same request.
    if let Ok(thinking) = std::env::var("AFFORDANCE_THINKING") {
        request["chat_template_kwargs"] = json!({"thinking": thinking == "1"});
    }
    request
}

#[test]
#[ignore = "needs AFFORDANCE_ROOT and AFFORDANCE_MODEL"]
fn affordance_race() {
    let race = Race::load();
    // Both sides loaded and warm before the clock starts.
    let warm = request(&[json!({"role": "user", "content": "warm up"})]);
    race.theirs(&warm);
    race.ours(&warm);

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(40);
    let mut turns = vec![json!({"role": "system", "content": "You are a careful coding assistant."})];
    let (mut theirs, mut ours, mut different, mut last) = (0u128, 0u128, 0, 0);
    let steps = 200;
    let mut first_difference = None;
    for i in 0..steps {
        turns.push(json!({"role": "user", "content": format!("Step {i}: {text}")}));
        turns.push(json!({"role": "assistant", "content": format!("Done with step {i}. {text}")}));
        let mut messages = turns.clone();
        messages.push(json!({"role": "user", "content": "continue"}));
        let request = request(&messages);
        let t = Instant::now();
        let want = race.theirs(&request);
        theirs += t.elapsed().as_nanos();
        let t = Instant::now();
        let got = race.ours(&request);
        ours += t.elapsed().as_nanos();
        if got != want {
            different += 1;
            if first_difference.is_none() {
                let at = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
                first_difference = Some(format!(
                    "step {i}: first differs at token {at} of {} (theirs) / {} (ours): theirs {:?} ours {:?}",
                    want.len(),
                    got.len(),
                    &want[at.saturating_sub(3)..(at + 6).min(want.len())],
                    &got[at.saturating_sub(3)..(at + 6).min(got.len())]
                ));
            }
        }
        last = want.len();
    }
    println!(
        "{}: SGLang with {}; {steps} requests growing to {last} tokens; different on {different}\n  \
         sglang {:.0} ms   affordance {:.0} ms   {:.1}x",
        race.model,
        race.kind(),
        theirs as f64 / 1e6,
        ours as f64 / 1e6,
        theirs as f64 / ours as f64,
    );
    let (render, tokenize) = (RENDER_NS.load(Ordering::Relaxed), TOKENIZE_NS.load(Ordering::Relaxed));
    if render + tokenize > 0 {
        println!("  of sglang's time: rendering text {:.0} ms, tokenizing it {:.0} ms", render as f64 / 1e6, tokenize as f64 / 1e6);
    }
    if let Some(difference) = first_difference {
        println!("  {difference}");
    }

    // Many conversations at once, their requests interleaved.
    let (conversations, steps) = (16, 40);
    let mut all: Vec<Vec<Value>> = (0..conversations)
        .map(|c| {
            vec![
                json!({"role": "system", "content": "You are a careful coding assistant."}),
                json!({"role": "user", "content": format!("Task {c}: start.")}),
            ]
        })
        .collect();
    let (mut theirs, mut ours, mut different_interleaved, mut last) = (0u128, 0u128, 0, 0);
    for i in 0..steps {
        for conversation in all.iter_mut() {
            conversation.push(json!({"role": "assistant", "content": format!("Done with step {i}. {text}")}));
            conversation.push(json!({"role": "user", "content": format!("Step {i}: {text}")}));
            let request = request(conversation);
            let t = Instant::now();
            let want = race.theirs(&request);
            theirs += t.elapsed().as_nanos();
            let t = Instant::now();
            let got = race.ours(&request);
            ours += t.elapsed().as_nanos();
            different_interleaved += usize::from(got != want);
            last = want.len();
        }
    }
    println!(
        "  {conversations} conversations interleaved, {steps} requests each, to {last} tokens; different on {different_interleaved}\n  \
         sglang {:.0} ms   affordance {:.0} ms   {:.1}x",
        theirs as f64 / 1e6,
        ours as f64 / 1e6,
        theirs as f64 / ours as f64,
    );

    // A tool-using agent: tools declared, calls whose arguments are a JSON string (as a client
    // sends them), results, answers.
    let tools = json!([
        {"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command."}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "read", "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    ]);
    let with_tools = |messages: &[Value]| {
        let mut request = request(messages);
        request["tools"] = tools.clone();
        request
    };
    let output = "src/main.rs\nsrc/lib.rs\nCargo.toml\n".repeat(30);
    let mut turns = vec![json!({"role": "system", "content": "You are a careful coding assistant."})];
    let (mut theirs, mut ours, mut different_tools, mut last) = (0u128, 0u128, 0, 0);
    let mut first_tool_difference = None;
    let steps = 60;
    for i in 0..steps {
        turns.push(json!({"role": "user", "content": format!("Step {i}: list the files, then read one.")}));
        turns.push(json!({"role": "assistant", "content": format!("Listing for step {i}."), "tool_calls": [{"id": format!("call_{i}_a"), "type": "function",
            "function": {"name": "bash", "arguments": json!({"command": format!("ls -la step{i}"), "timeout": 30}).to_string()}}]}));
        turns.push(json!({"role": "tool", "tool_call_id": format!("call_{i}_a"), "content": output}));
        turns.push(json!({"role": "assistant", "content": format!("Step {i} is done.")}));
        let mut messages = turns.clone();
        messages.push(json!({"role": "user", "content": "continue"}));
        let request = with_tools(&messages);
        let t = Instant::now();
        let want = race.theirs(&request);
        theirs += t.elapsed().as_nanos();
        let t = Instant::now();
        let got = race.ours(&request);
        ours += t.elapsed().as_nanos();
        if got != want {
            different_tools += 1;
            if first_tool_difference.is_none() {
                let at = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
                let near = |ids: &[u32]| {
                    race.tokenizer
                        .decode(&ids[at.saturating_sub(12)..(at + 14).min(ids.len())], false)
                        .map(|d| format!("{d:?}"))
                        .unwrap_or_default()
                };
                first_tool_difference = Some(format!("step {i}: differs at token {at}\n    theirs: {}\n    ours:   {}", near(&want), near(&got)));
            }
        }
        last = want.len();
    }
    println!(
        "  a tool-using agent, {steps} requests growing to {last} tokens; different on {different_tools}\n  \
         sglang {:.0} ms   affordance {:.0} ms   {:.1}x",
        theirs as f64 / 1e6,
        ours as f64 / 1e6,
        theirs as f64 / ours as f64,
    );
    if let Some(difference) = first_tool_difference {
        println!("  {difference}");
    }

    // What the view does not render is the model's own formatter's: the same ids, by construction.
    let continued = {
        let mut request = request(&[json!({"role": "user", "content": "count to five"}), json!({"role": "assistant", "content": "one, two,"})]);
        request["continue_final_message"] = json!(true);
        request
    };
    assert_eq!(
        race.ours.encode_request(continued.clone(), &HashMap::new()).map(|ids| ids.ok()),
        race.theirs.encode_request(continued, &HashMap::new()).map(|ids| ids.ok()),
        "a continued final message goes to the model's own formatter"
    );
    assert_eq!(different + different_interleaved + different_tools, 0, "the two formatters must produce the same ids");
}

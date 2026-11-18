# SGLang with Affordance building its prompts

This is [SGLang](https://github.com/sgl-project/sglang) 0.5.23, a model server, with one change.
When a chat request arrives, SGLang's Rust server turns its messages into the tokens the model
reads: it fills in the model's chat template, then splits the text into tokens. On this branch
[Affordance](https://github.com/approximatelabs/affordance) builds those tokens instead, from a
template app written in TSX. It keeps the tokens of the turns it has already built, so a request
in a long conversation only builds its new turn.

On Qwen3.6-35B-A3B, with prompts over 120,000 tokens, the first token of a reply came after 552 to
562 ms with stock SGLang and 421 to 422 ms with this branch, over three runs each. On 2,703 test
requests, including ones with tools, the server produced the same tokens with Affordance as
without it. The measurements are on
[affordance.rs/showcase/first-token](https://affordance.rs/showcase/first-token).

## Running it

Build the Rust server as upstream does (`rust/sglang-server`). Then name an Affordance template
app for the model with three environment variables. The apps are in the Affordance repository,
listed by model in `views/index.json`. For Qwen3.6-35B-A3B:

```sh
git clone https://github.com/approximatelabs/affordance
export SGLANG_AFFORDANCE_VIEW=$PWD/affordance/views/third_party/qwen/Qwen3.6-35B-A3B.tsx
export SGLANG_AFFORDANCE_ENTRY=Main
export SGLANG_AFFORDANCE_TOKENIZER=/path/to/Qwen3.6-35B-A3B/tokenizer.json
```

With the three set, the server logs that chat requests render through the Affordance app. A
request the app does not cover goes to SGLang's own formatter, as without this branch.

The change is in `rust/sglang-processor/src/render/` (`selection.rs` and `mod.rs`). The race
against SGLang's own formatter is `rust/sglang-processor/tests/affordance_race.rs`.

This branch is the code behind the measurements above, kept on SGLang 0.5.23. It does not follow
SGLang's releases.

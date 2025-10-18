Thanks for checking out OpenArc!

## Motivation

My goal with this project has been to make using Intel devices as accelerators easier. 

That's it. 

In the tradition of `llama.cpp`, OpenArc tries to enable building cool stuff on top of a system that *just works* while being hackable. 

## Codebase

Our objective is simple; stay as close to standard python library as possible. 

We don't use a database, do not persist AI artifacts and are not interested in opinionated pre or postprocessing; instead, we focus on the part of inference-work that really matters; ingesting data models can consume, and outputting data that programs can consume. 

### [server.py](src/server/server.py)

- Uses `fastapi`
- Orchestrates all incoming requests
- Maintains OpenAI compatability
- Mostly standard library

### [worker_registry.py](src/server/worker_registry.py) and [model_registry.py](src/server/model_registry.py)

- Manages model lifecycle in memory using asyncio
- Worker pattern
    - Each model gets an inference worker, which processes requests
    - Each model gets a queue worker, which sequentially delivers request contents

### Inference Pipelines

This is where the standard library/minimal dependency approach becomes less strict. 

We want to support as many thoughtfully implemented tasks as people find useful. `server` and `worker` layer can orchestrate *any* inferencing work you can come up with and over time they will continue to improve. 

If that seems useful to you, here's how you can contribute.


## Contributing, broadly

1. Open an issue describing the change you want to make.
2. Fork the repository
3. Create a new branch
4. Make your changes
5. Open a pull request and fill out the template

## PR Template

```
## What does this PR do

## Usage/Examples

## Discussion

```

OpenArc's PR template encourages thoughtful introspection on work you contribute. 
PRs which do maintenence 

### What does this PR do

- describe the goal of your changes and what they do.
- Be concise, and think about the big picture of OpenArc.
- Generally PRs should have narrow scope; try not to change the codebase to much when introducing. If you find something should be refactored, or improved to make your changes, that a sign you need more than one PR. 
- If you increased performance, make sure you have measured it

### Usage

- Provide the simplest possible way to use every feature you have added.
- This makes review easier, and is self documenting. 
- Expect your examples to be used in the documentation, and as tests during review.

### Discussion

- Offer motivation and discuss your strategy
- Take time to walk through challenges you faced, or issues you solved while working on the PR.
- Reference documentation, papers, code examples discussions on forums- anything you found which helps the reviewer understand what












## Code Quality and AI Tools

I have learned coding using AI tools and am not here to judge. Still, we need to define a standard.

AI tools are a fantastic resource, especially for learning more about AI- most of this project was written in natural language and pseudocode with turn-based editing and agents using Cursor. Tab. Tab. Tab. 

That said, I review my own generated code line by line to ensure correctness and expect the same level of scrutiny from anyone who contributes to OpenArc. Nothing disqualifies a PR from review, but I ask that you respect my time and everyone elses by working to limit slop. 






- Changes should be related to your PR. 
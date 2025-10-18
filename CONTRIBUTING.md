Thanks for checking out OpenArc!

## Philosophy

My goal with this project has been to make using Intel devices as accelerators for inference easier. 

That's it. 

Over time I will add more tasks and model implementations as I learn more- these will eventually extend beyond OpenVINO.

## Codebase

Our objective is simple; stay as close to standard python library as possible. 

We don't use a database, do not persist AI artifacts and are not interested in opinionated pre or postprocessing. In the tradition of `llama.cpp`, OpenArc tries to enable building cool stuff on top of a system that *just works* in an area of the inference stack where things do not usually *just work*. 

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

We want to support as many thoughtfully implemented tasks as people find useful. My intention with `server` and `worker` layer 


## Contributing, broadly

1. Open an issue describing the change you want to make.
2. Fork the repository
3. Create a new branch
4. Make your changes
5. Open a pull request and fill out the template

## PR Template

OpenArc's PR template encourages thoughtful introspection on work you contribute. 

```
## What does this PR do

## Usage/Examples

## Discussion

```


### What does this PR do

- describe the goal of your changes and what they do.
- Be concise, and think about the big picture of OpenArc.

### Usage

- Provide the simplest possible way to use every feature you have added.
- This makes review easier, and is self documenting. 
- Code communciates a lot, but people help connect concepts together. Break high level concepts down 

### Discussion

- offer motivation and discuss your strategy
- Take time to walk through challenges you faced, or issues you solved while working on the PR.
- reference documentation, papers, discussions on forums- anything to spark quality
- 




```
## What does this PR do

## Usage/Examples

## Discussion

```







## Code Quality and AI Tools

I have learned coding using AI tools and am not here to judge. Still, we need to define a standard.

AI tools are a fantastic resource, especially for learning more about AI- most of this project was written in natural language and pseudocode in turn-based editing and agents using Cursor. Tab. Tab. Tab.

That said, I review my own generated code line by line to ensure correctness and expect the same level of scrutiny from anyone who contributes to OpenArc. Nothing disqualifies a PR from review, but I ask that you respect my time and everyone elses by working to limit slop. 

### Ground rules for generated code

> [!Note] Check back here before you mark anything as ready for review.
> There are many types of slop,

- Generated omments are not allowed.
- Code documentation should not mention changes.
- Methods, functions, classes snippets should not contain language from instructions.
- 




A solid PR

- Adds something to the project
- Implements a paper
- Refines existing systems

- Maintains literally anything





- Changes should be related to your PR. 
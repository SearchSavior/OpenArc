### Errors

OpenArc uses arguments from ```openarc add``` as metadata to make routing decisions; think of it like you are querying for inference code. 

When an ```openarc load``` command fails, the CLI tool displays the full stack trace to help you figure out why.

### Model concurrency

More than one model can be loaded into memory at once in a non blocking way. 

Each model gets its own first in, first out queue, scheduling requests based on when they arrive. Inference of one model can fail without taking down the server, and many inferences can run at once.

However, OpenArc *does not batch requests yet*. Paged attention-like continuous batching for ```llm``` and ```vlm``` will land in a future release.
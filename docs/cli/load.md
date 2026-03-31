# openarc load

After using `openarc add` you can use `openarc load` to read the added configuration and load models onto the OpenArc server.

OpenArc uses arguments from `openarc add` as metadata to make routing decisions internally; you are querying for correct inference code.

## Load a single model

```
openarc load <model-name>
```

## Load multiple models at once

```
openarc load <model-name1> <model-name2> <model-name3>
```

Be mindful of your resources; loading models can be resource intensive! On the first load, OpenVINO performs model compilation for the target `--device`.

When `openarc load` fails, the CLI tool displays a full stack trace to help you figure out why.

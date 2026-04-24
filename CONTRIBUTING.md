## Thanks for checking out OpenArc!

The goal with this project has been to make using Intel devices as accelerators easier. 

## Contributing

- Open an issue before beginning work 

## Guidelines

- Create separate PRs for each feature or fix. Avoid combining unrelated changes in a single PR
- Consider allowing write access to your branch for faster reviews, as reviewers can push commits directly

## Docs

OpenArc's documentation website uses a python library called [Zensical](https://zensical.org/docs/get-started).

To edit the docs and view changes in real time use

```
zensical serve -a localhost:8004 # build and serve the docs locally
```

and make changes to markdown files in `docs/`.  

### Github Actions

The docs site is configured to rebuild any time there are changes to the `docs/` directory in a PR so once a PR is merged, the site is rebuild with new changes. Very nice.


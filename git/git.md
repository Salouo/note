## 1. Upload my branch

1. Switch to `chen` branch.

   ```sh
   git checkout chen
   ```

   

2. Check status of modifications

   ```sh
   git status
   ```

   

3. Stage the modifications of `chen/`.

   ```sh
   git add chen/
   ```



4. Commit a comment.

   ```sh
   git commit -m 'update'
   ```



5. Push the local submission to remote repository.

   ```sh
   git push origin chen
   ```

   

## 2. Git commit types

| Type     | Purpose                                 | Example                                  |
| -------- | --------------------------------------- | ---------------------------------------- |
| feat     | New feature                             | `feat(api): add /v1/upload endpoint`     |
| fix      | Bug fix                                 | `fix(parser): handle empty line`         |
| docs     | Docs / Comments                         | `docs(readme): add quickstart`           |
| style    | Code formatting (no logical changes)    | `style: format with black`               |
| refactor | Refactor (no new features, no bug fixs) | `refactor(core): simplify loop`          |
| perf     | Performance improvements                | `perf(db): reduce query count`           |
| test     | Tests / Tests code                      | `test: add unit tests for scorer`        |
| build    | Build / packaging / release scripts     | `build: add uv lock file`                |
| ci       | CI/CD configuration                     | `ci: cache pip wheels`                   |
| deps     | Dependency upgrade                      | `deps: bump numpy to 2.0`                |
| chore    | Chore / misc (no functional impact)     | `chore: rename project folders`          |
| revert   | Revert commit                           | `revert: feat(api): add upload endpoint` |

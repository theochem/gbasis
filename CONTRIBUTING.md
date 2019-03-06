# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, Slack, Trello, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build.
2. Update the README.md with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this
   Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. You may merge the Pull Request in once you have the sign-off of two other developers, or the
   project maintainer.

## Step By Step Workflow
NOTE: This workflow pertains specifically to the developers at the Pre-Alpha and Alpha stages of the
package.
1. Clone the main repository.
```bash
git clone repository_link origin
```
2. Fork the main repository. The main repository will be push protected, which means that you cannot
   contribute to it without making pull requests. Add forked repository to your remote.
```bash
git remote add pullrequest forked_repository_link
```
   Name the repository however you'd like. One name is "pullrequest"  and another is the name of the
   owner of the repository.
3. Periodically update the main repository.
```bash
git fetch origin
```
4. Assign yourself a card on the Trello board (https://trello.com/b/xLPzbh3M/gbasis) in the `To Do`
   section. Move the card to the `In Progress` section.
5. Make a branch from the (updated) master branch.
```bash
git branch new_branch origin/master
git checkout new_branch
```
6. Work on your task. Try to make appropriate sized commits with informative messages.
7. Push your branch to your (forked) repository. Try to push whenever you make a commit.
```bash
git push pullrequest new_branch
```
8. Make a pull request from the pushed branch of the forked repository to the master branch of the
   main repository (`Pull Requests tab > New Pull Request`). The reason you are making an unfinished
   pull request is so that you inform others of your progress. Of course, you are free to keep your
   branch private, but you will eventually need to push to contribute to the main repository. There
   *should* be no judgement for incomplete code, but if you do feel uncomfortable with pushing a
   work in progress, let the appropriate supervisor know.
9. Add the Trello Card link (`Actions > Share > Link`) as a comment to the pull request.
10. Link the Trello Card to the pull request (`Power-Ups > GitHub > Attach Pull Request`)
11. Continue working on task. Push to your forked repository whenever you can.
12. Once you have finished the task, move your card to the `Quality  Assurance` section. This will
    let others know that you're ready to get your pull request reviewed. Before you do, however, it
    is a good idea to make sure that your pull request passes all essential QA (quality assurance)
    checks. We currently use TravisCI to handle the QA checks. If your pull request does not pass
    the QA, go to `show all checks > details > The build > job id` to see the log of the QA tests.
    You can also run these tests locally by going to the parent directory of the package and
```bash
tox
```
13. Wait for reviews. If the reviewer (usually the project maintainer) is not responsive, send
    messages to one or more of the following: GitHub Pull Request Thread, Trello, Slack, and Email.
    If they are still not responsive, contact the "next one up".
14. Make appropriate changes according to the reviewer's comments. Feel free to ask for their
    reasoning. It is the responsibility of the reviewer to inform you of their review.
15. Once your pull request has been approved by the reviewers, you update the master branch on the
    main repository,
```bash
git fetch origin
```
    (interactively) rebase your branch on top of the master on the main repository,
```bash
git checkout new_branch
git rebase -i origin/master
```
    Squash commits that are redundant, polish up your commit messages, and organize your commits to
    be somewhat intuitive. If you are unsure what this means, ask the person that will be accepting
    your pullrequest.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers
pledge to making participation in our project and our community a harassment-free experience for
everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level
of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic address, without explicit
  permission
* Other conduct which could reasonably be considered inappropriate in a professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are
expected to take appropriate and fair corrective action in response to any instances of unacceptable
behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits,
code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or
to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is
representing the project or its community. Examples of representing a project or community include
using an official project e-mail address, posting via an official social media account, or acting as
an appointed representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting
the project team at EMAIL. All complaints will be reviewed and investigated and will
result in a response that is deemed necessary and appropriate to the circumstances. The project team
is obligated to maintain confidentiality with regard to the reporter of an incident. Further details
of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face
temporary or permanent repercussions as determined by other members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4, available at
[http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/

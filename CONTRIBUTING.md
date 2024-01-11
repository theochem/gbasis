# Contributing to gbasis
Hello! Thank you for taking your time to contribute!

Please note we have a [Code of Conduct](https://qcdevs.org/guidelines/QCDevsCodeOfConduct/) , please follow it in all your interactions with the project. Please report unacceptable behavior to [theochem@github.com](mailto:theochem@github.com).

## Do you have a question?

* Check if the question has been asked by searching on GitHub under  [Issues: Question](https://github.com/theochem/gbasis/issues?utf8=%E2%9C%93&q=is%3Aissue+%5BQUESTION%5D+).

* If you're unable to find a similar question, [open a new one](https://github.com/theochem/gbasis/issues/new/choose) under Question.

* Please note that questions may take a long time until they get answered, if at all.

## Did you find a bug?

* **Ensure that the bug was not already reported** by searching on GitHub under [Issues: Bugs](https://github.com/theochem/gbasis/labels/bug).

* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/theochem/gbasis/issues/new/choose) under Bug Report. Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

## Are you requesting a feature?

* **Ensure that the feature is not already requested** by searching on GitHub under [Issues: Features](https://github.com/theochem/gbasis/labels/enhancement).

* If you're unable to find an open issue for the feature, [open a new one](https://github.com/theochem/gbasis/issues/new/choose) under Feature Request. Be sure to include a **title and clear description**, as much relevant information as possible, and describe the desired behaviour.

## Did you write a patch for a bug or add a new feature?

* Open a new GitHub pull request.

* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

* Before submitting, please read the [Pull Request Guideline](#pull-request-guideline).

## Did you fix whitespace, format code, or make cosmetic changes?

Feel free to make a pull request (see [Pull Request Guideline](#pull-request-guideline)) with a title that starts with [COSMETIC]. Please note that cosmetic pull requests may be rejected with little explanation. Reasoning behind these decisions can be explained [here](https://github.com/rails/rails/pull/13771#issuecomment-32746700).

## Pull Request Guideline

* Try to keep your commits clean. This involves squashing commits together that do basically the same thing (e.g. fix problem, fix problem1, ...) or are cosmetic (e.g. fix typo), dividing a large commit into multiple smaller commits, and dropping commits that are not relevant to the PR. It is easier to squash commits together than it is to divide them, so it's normally good practice to keep your commits small.

* Rebase your changes to the latest master.

* Ensure that all of the contributed code is unit tested (code is tested in bite size pieces rather than at the end), passes coverage (every part of the code is tested), and follows the styleguide. To check, install tox (`pip install --user tox`) and run `tox -e qa` from the project home directory.

* Update the README.md if necessary (e.g. new feature is added).

* Increase the version numbers in `setup.py` and `__version__.py` to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).

* The Pull Request will be merged in once you have the sign-off of two other developers, or the project maintainer.

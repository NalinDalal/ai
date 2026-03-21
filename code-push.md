# Code Push & Repo Split Guide

This file documents the process for splitting folders into standalone repos while preserving commit history, and exporting commit logs for reference. Use this workflow for future projects and papers.

### 1. Stage and commit

### 2. Split a Folder into a New Repo (Preserve History)
- Use `git subtree split` to create a branch with the history of a specific folder:
  ```
  git subtree split --prefix=<folder> -b <branch-name>
  ```
  Example:
  ```
  git subtree split --prefix=week-1-2 -b week-1-2-branch
  ```

### 3. Add New Remote Repo
- Add the new repo as a remote:
  ```
  git remote add <remote-name> <repo-url>
  ```
  Example:
  ```
  git remote add python-basics git@github.com:NalinDalal/python-basics.git
  ```

### 4. Push Split Branch to New Repo
- Push the split branch to the new repo's main branch:
  ```
  git push <remote-name> <branch-name>:main
  ```
  Example:
  ```
  git push python-basics week-1-2-branch:main
  ```

### 5. Export Commit Log for Reference
- Save the commit history for the folder to a text file:
  ```
  git log --oneline -- <folder> > <folder>-log.txt
  cp <folder>-log.txt <folder>/
  ```

---

## Why Do This?
- Keeps each project/paper organized and independent
- Preserves full development history for transparency and learning
- Makes it easy to link, share, and showcase work

---

## Repeat for Any Folder
- Just change `<folder>`, `<branch-name>`, `<remote-name>`, and `<repo-url>` as needed.

---

## Example Repos
- python-basics (week-1-2)
- baisc-papers (week-3)
- machine-learning (week-3-ml)
- neural-networks (week-7)
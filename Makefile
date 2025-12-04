# Simple Makefile for local workflows

SHELL := /bin/sh

APP_NAME       := fraud-detection
DEV_IMAGE      := $(APP_NAME)-dev
PROD_IMAGE     := $(APP_NAME)-prod
DEV_CONTAINER  := $(APP_NAME)-dev-container

# ----- Python tooling -----

uv:
	uv sync --extra dev

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy internalpy api model

test:
	uv run pytest

check: lint typecheck test ## Run all checks


# ----- Docker: local dev (hot reload via Dockerfile.dev) -----

build:
	docker build -f Dockerfile.dev -t $(DEV_IMAGE) .

run:
	-docker stop $(DEV_CONTAINER) 2>/dev/null || true
	-docker rm $(DEV_CONTAINER) 2>/dev/null || true
	docker run \
		--name $(DEV_CONTAINER) \
		--network host \
		-v ${PWD}:/app \
		-v $$HOME/.config/gcloud:/root/.config/gcloud \
		--env-file .env \
		$(DEV_IMAGE)

shell:
	docker exec -it $(DEV_CONTAINER) /bin/sh


# ----- Docker: prod build (for local sanity testing) -----

buildp:
	docker build -f Dockerfile -t $(PROD_IMAGE) .

runp:
	-docker stop $(DEV_CONTAINER) 2>/dev/null || true
	-docker rm $(DEV_CONTAINER) 2>/dev/null || true
	docker run \
		--name $(DEV_CONTAINER) \
		-p 8080:8080 \
		-e PROJECT_ID=$(PROJECT_ID) \
		-e PLATFORM=loc \
		$(PROD_IMAGE)


# ----- Utilities -----

clean-docker:
	docker system prune -f


# =====================================================================
#                           GIT HELPERS
# =====================================================================

# Simple one-shot commit & push in the current repo:
#   make git "Your commit message"
#
# If there is nothing staged, it will just say so and skip the commit.

.PHONY: git

git:
	@echo "MAKECMDGOALS = $(MAKECMDGOALS)"
	@echo "msg = $(msg)"
	@if [ -z "$(msg)" ]; then \
		echo "Error: Commit message not provided. Use: make git \"Your commit message\""; \
		exit 1; \
	fi
	git add .
	@if ! git diff --cached --quiet --exit-code; then \
		git commit -m "$(msg)"; \
	else \
		echo "Nothing to commit — skipping git commit"; \
	fi
	git push origin

# `msg` is everything after the 'git' target
msg := $(strip $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)))


# =====================================================================
#                    WORKTREE-BASED FLOW (OPTIONAL)
# =====================================================================
# This lets you work in a separate directory per branch so you *can't*
# "forget to switch branches":
#
#   - MAIN_DIR  = main repo root (shared git dir parent)
#   - WORK_BRANCH (default: "work") is the branch for the worktree
#   - WORK_DIR  = sibling folder where the worktree lives
#
# Typical flow:
#   make winit         # create a worktree for branch 'work'
#   cd <printed WORK_DIR>  # start working there
#   make wbootstrap    # (from inside worktree) install dev deps & pre-commit
#   make wcom "msg"    # commit in worktree with pre-commit
#   make wsync         # rebase worktree onto origin/main
#   make wmerge        # fast-forward main to worktree and push

# Figure out the "main" repo dir even when we're in a worktree:
# git-common-dir is something like /path/to/fraud-detection/.git
GIT_COMMON_DIR := $(shell git rev-parse --git-common-dir 2>/dev/null)
MAIN_DIR       := $(abspath $(GIT_COMMON_DIR)/..)

# Branch name for the worktree (override with WORK_BRANCH=my-feature)
WORK_BRANCH ?= work

# Default worktree directory: <main-dir>-<branch>
WORK_DIR    ?= $(abspath $(MAIN_DIR)-$(WORK_BRANCH))

# If we are *already* inside the worktree dir (e.g. fraud-detection-work),
# prefer the current directory as WORK_DIR. This fixes the "-work-work" bug.
ifeq ($(abspath $(CURDIR)),$(WORK_DIR))
override WORK_DIR := $(abspath $(CURDIR))
endif

.PHONY: winit wstat wbootstrap wcom wsync wmerge

winit:
	@echo "→ creating worktree '$(WORK_BRANCH)' at $(WORK_DIR)…"
	@git -C "$(MAIN_DIR)" fetch origin >/dev/null 2>&1 || true
	@if [ -d "$(WORK_DIR)" ]; then \
	  echo "worktree already exists at $(WORK_DIR)"; \
	else \
	  if git -C "$(MAIN_DIR)" show-ref --verify --quiet refs/remotes/origin/main; then \
	    git -C "$(MAIN_DIR)" worktree add "$(WORK_DIR)" -b "$(WORK_BRANCH)" origin/main; \
	  else \
	    git -C "$(MAIN_DIR)" worktree add "$(WORK_DIR)" -b "$(WORK_BRANCH)" main; \
	  fi; \
	fi
	@echo "→ cd $(WORK_DIR)    # start coding there"

wstat:
	@echo "== main ($(MAIN_DIR)) ==";  git -C "$(MAIN_DIR)" status -sb
	@echo ""; echo "== $(WORK_BRANCH) ($(WORK_DIR)) =="; \
	  if [ -d "$(WORK_DIR)" ]; then \
	    git -C "$(WORK_DIR)" status -sb; \
	  else \
	    echo "[no worktree dir yet at $(WORK_DIR)]"; \
	  fi

wbootstrap:
	@echo "→ installing dev deps in $(WORK_DIR)…"
	@cd "$(WORK_DIR)" && uv sync --extra dev && uv run pre-commit install || true

# Capture commit message for worktree commits
CMSG := $(strip $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)))
COMMIT_MSG ?= $(CMSG)

wcom:
	@if [ -z "$(COMMIT_MSG)" ]; then \
	  echo "Usage: make wcom \"Your commit message\"  (or COMMIT_MSG=…)"; exit 1; fi
	@cd "$(WORK_DIR)" && git add -A
	@cd "$(WORK_DIR)" && uv run pre-commit run --all-files || true
	@cd "$(WORK_DIR)" && if ! git diff --cached --quiet --exit-code; then \
	  git commit -m "$(COMMIT_MSG)"; \
	  echo "✓ committed to $(WORK_BRANCH): $(COMMIT_MSG)"; \
	else echo "Nothing to commit in $(WORK_DIR)"; fi

wsync:
	@echo "→ rebasing $(WORK_BRANCH) worktree onto latest origin/main…"
	@git -C "$(WORK_DIR)" fetch origin >/dev/null 2>&1 || true
	@cd "$(WORK_DIR)" && git rebase origin/main || true

wmerge:
	@echo "→ fast-forwarding main to $(WORK_BRANCH) and pushing…"
	@git -C "$(MAIN_DIR)" fetch origin >/dev/null 2>&1 || true
	@git -C "$(WORK_DIR)" fetch origin >/dev/null 2>&1 || true
	@cd "$(WORK_DIR)" && git rebase origin/main
	@cd "$(MAIN_DIR)" && git checkout main >/dev/null 2>&1 || true
	@cd "$(MAIN_DIR)" && git merge --ff-only "$(WORK_BRANCH)" || { \
	  echo "✗ fast-forward merge failed. Resolve conflicts or rebase $(WORK_BRANCH) first."; exit 1; }
	@cd "$(MAIN_DIR)" && git push origin main
	@echo "✓ merged and pushed main"
%:
	@:
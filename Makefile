.PHONY: install install-hpo-rl-bench install-other-dependencies help
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

install: install-hpo-rl-bench install-other-dependencies
install-hpo-rl-bench: ## run tests quickly with the default Python
	git submodule update --init --recursive
	pip install -r hpo_rl_requirements.txt
install-other-dependencies:
	pip install arlbench gymnasium==0.29.1 xminigrid==0.8.0 tqdm
	pip install "hypersweeper[dehb]"
	pip install numpy==1.24.0 configspace==0.6.1 smac==2.0.2 gpytorch rich
	

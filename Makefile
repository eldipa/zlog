all:
	@echo "Nothing to compile, enjoy but remember this is Proof of Concept!"

test:
	@hash byexample 2>/dev/null || echo "byexample is missing; install it with 'pip install byexample'"
	@hash byexample 2>/dev/null
	@byexample -l python --ff zlog.py

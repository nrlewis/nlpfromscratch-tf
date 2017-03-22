all: 
	echo "all called" 

.PHONY: test

test:
	python -m unittest discover test

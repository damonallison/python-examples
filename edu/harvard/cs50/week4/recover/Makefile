.PHONY: recover

recover:
	clang *.c -lcs50 -fsanitize=address -o recover

go: recover
	./recover

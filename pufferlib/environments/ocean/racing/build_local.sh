# #!/bin/bash

# clang -Wall -Wuninitialized -Wmisleading-indentation -fsanitize=address \
#     -ferror-limit=3 -g -o racinggame racing.c \
#     -I./raylib-5.0_linux_amd64/include/ \
#     -L./raylib-5.0_linux_amd64/lib/ -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 \
#     -DPLATFORM_DESKTOP

gcc -c pufferlib/environments/ocean/racing/racing.c -o pufferlib/environments/ocean/racing/racing.o -I/path/to/python/include -I/path/to/numpy/include
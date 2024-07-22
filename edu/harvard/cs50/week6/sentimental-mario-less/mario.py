"""
Creates a half-pyramid using hashes for blocks

   #
  ##
 ###
####
"""


def half_pyramid(height: int) -> None:
    max_height = height - 1
    for row in range(height):
        for col in range(height):
            if col >= max_height - row:
                print("#", end="")
            else:
                print(" ", end="")
        print("")


if __name__ == "__main__":
    while True:
        try:
            height = int(input("Height: "))
            if height < 1 or height > 8:
                continue
            break
        except Exception as e:
            print(f"Exception: {e}")

    half_pyramid(height)

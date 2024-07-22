def min_coins(cents: int) -> int:
    """Returns the minimum number of coins which totals `cents`."""
    coins = 0
    while cents > 0:
        if cents >= 25:
            cents -= 25
        elif cents >= 10:
            cents -= 10
        elif cents >= 5:
            cents -= 5
        else:
            cents -= 1
        coins += 1
    return coins


if __name__ == "__main__":
    while True:
        try:
            # assumes the user is going to enter the amount in dollars.
            #
            # i.e., 1 == 1 dollar
            amount = float(input("Change: "))
            if amount < 0:
                continue
            amount = int(amount * 100)
            break
        except Exception as e:
            print("enter a valid dollar amount. example: 5 or 5.00")

    print(min_coins(amount))

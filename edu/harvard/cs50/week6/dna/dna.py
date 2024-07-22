import csv
import sys


def main():

    if len(sys.argv) != 3:
        print("usage: dna <database.csv> <sequences.txt>")
        return
    db_file = sys.argv[1]
    seq_file = sys.argv[2]

    people: list[dict[str, str]] = []
    with open(db_file) as f:
        db = csv.DictReader(f)
        # assumes the first column in db is the person's name
        dna_seq_names: list[str] = db.fieldnames[1:]

        for row in db:
            people.append(row)

    with open(seq_file, "r") as f:
        full_seq = f.read()

    # Find longest match of each STR in DNA sequence
    matches: dict[str, int] = {}
    for seq_name in dna_seq_names:
        matches[seq_name] = longest_match(full_seq, seq_name)

    # Check database for matching profiles
    for person in people:
        found = True
        for k, v in matches.items():
            if int(person[k]) != v:
                found = False
                break

        if found:
            print(person["name"])
            return
    print("No match")


def longest_match(sequence, subsequence):
    """Returns length of longest run of subsequence in sequence."""

    # Initialize variables
    longest_run = 0
    subsequence_length = len(subsequence)
    sequence_length = len(sequence)

    # Check each character in sequence for most consecutive runs of subsequence
    for i in range(sequence_length):

        # Initialize count of consecutive runs
        count = 0

        # Check for a subsequence match in a "substring" (a subset of characters) within sequence
        # If a match, move substring to next potential match in sequence
        # Continue moving substring and checking for matches until out of consecutive matches
        while True:

            # Adjust substring start and end
            start = i + count * subsequence_length
            end = start + subsequence_length

            # If there is a match in the substring
            if sequence[start:end] == subsequence:
                count += 1

            # If there is no match in the substring
            else:
                break

        # Update most consecutive matches found
        longest_run = max(longest_run, count)

    # After checking for runs at each character in seqeuence, return longest run found
    return longest_run


main()

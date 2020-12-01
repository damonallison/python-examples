import unittest


class ListTests(unittest.TestCase):
    """Python's list() is an ordered, mutable data structure
       which can elements of different types."""

    def test_mutability(self) -> None:
        """Lists are mutable "reference" types"""

        a = [1, 2]
        b = a
        a[0] = 3

        assert a == b
        assert 3 == b[0]


    def test_copy(self):
        """Lists are ordered, mutable.

        Lists are not strongly typed. Lists can contain elements
        of multiple types.

        To copy, use [:]. Always use [:] for iteration when you are
        modifying the list.
        """

        lst = ["damon", "kari", 10, ["another", "list"]]
        copy = lst.copy()

        self.assertFalse(lst is copy,
                         msg="The objects should not be referentially equal.")
        self.assertEqual(lst, copy,
                         msg="The lists should be logically (==) equal.")

        #
        # Copies are shallow.
        #
        # Here, we are copying value types. lst2 will have a separate copy of
        # each element.
        #
        lst = [1, 2]
        copy = lst.copy()

        copy[0] = 3
        self.assertListEqual([1, 2], lst)
        self.assertListEqual([3, 2], copy)

        #
        # Here, we are copying reference types. lst2 will contain a pointer to
        # the same lst[0] element.
        lst = [[1], 2]
        copy = lst.copy()
        copy[0][0] = 3
        self.assertEqual([[3], 2], lst)

    def test_list_sorting(self):
        """Example showing max(), min(), sorted()

        max() retrieves the max element (as defined by >)
        min() retrieves the min element (as defined by <)
        sorted() will sort according to < and return a *new* list.
        """

        a = [10, 20, 1, 2, 3]

        self.assertEqual(20, max(a))
        self.assertEqual(1, min(a))
        self.assertEqual([1, 2, 3, 10, 20], sorted(a))
        self.assertEqual([20, 10, 3, 2, 1], sorted(a, reverse=True))
        # Sorted returns a copy
        self.assertEqual([10, 20, 1, 2, 3], a)

        b = a.copy()

        a.sort()  # sort() will sort in place.
        self.assertEqual(a, sorted(b))
        self.assertEqual([10, 20, 1, 2, 3], b)

    def test_list_append(self):
        """Use .append(elt) and .extend(iterable) to append to a list."""

        # Append will add a single value.
        lst = ["damon"]
        lst.append(42)

        # + will concatentate the two lists (use extend() instead for clarity)
        lst = lst + ["cole", 11]
        expected = ["damon", 42, "cole", 11]

        self.assertEqual(expected, lst)

        # .extend(iterable) will append all items from iterable.
        # This is preferred over using `+` since it's clear what
        # you expect.
        expected.extend(["grace", 13])
        lst.extend(["grace", 13])
        self.assertEqual(expected, lst)

    def test_list_joining(self):
        """Joining allows you to combine lists of strings"""

        names = ["damon", "ryan", "allison"]
        self.assertEqual(" ".join(names), "damon ryan allison")

    def test_iteration(self):
        """`for` iterates over the elements in a sequence."""

        lst = ["damon", "kari", "grace", "lily", "cole"]
        expected = []

        # Remember to always iterate over a *copy* of the list
        for name in lst.copy():
            expected.append(name)
        self.assertEqual(expected, lst)

        # To iterate over the indices of a sequence, use range(len(lst))
        expected = []
        for i in range(len(lst)):
            expected.append(lst[i])
        self.assertEqual(expected, lst)

        #
        # To iterate over indices and values simultaneously, use enumerate()
        # enumerate() returns tuples of the indicies and values of a list.
        #
        pos = []
        val = []
        for i, v in enumerate(["tic", "tac", "toe"]):
            pos.append(i)
            val.append(v)

        self.assertEqual([0, 1, 2], pos)
        self.assertEqual(["tic", "tac", "toe"], val)

        # Loop statements can have an `else` clause, which executes
        # when the loop terminates without encoutering a `break` statement
        primes = []
        for n in range(2, 6):
            for x in range(2, n):
                if n % x == 0:
                    break
            else:
                primes.append(n)
        self.assertEqual([2, 3, 5], primes)

    def test_zip(self):
        """zip() returns an iterator that combines multiple iterables
        into a sequence of tuples.

        Each tuple contains the elements in that position from all the
        iterables.

        Once one list is exhaused, zip stops.

        The object that zip() returns is a zip() object.

        See:
        https://docs.python.org/3.3/library/functions.html#zip

        """

        # Here are two iterables that we are going to pass to zip().
        # Notice how ages only has 2 elements. This will force zip() to stop
        # after two elements, leaving "joe" out of the resulting zip() result.

        # If you care about the trailing, unmatched values from longer
        # iterables, use itertools.zip_longest() instead.

        names = ("damon", "jeff", "joe")
        ages = [20, 32]

        # zip() will return a zip() object, which is an iterator. You need to
        # cast the iterator into a concrete type (tuple, list, set, dict) to
        # realize the iterable.

        self.assertEqual((("damon", 20), ("jeff", 32)),
                         tuple(zip(names, ages)))
        self.assertEqual([("damon", 20), ("jeff", 32)],
                         list(zip(names, ages)))

        # You can use * to "unzip" a list or tuple. In this case, we will unzip
        # people to send zip() the inner tuples.
        people = (("damon", "jeff"), (20, 32))
        self.assertEqual((("damon", 20), ("jeff", 32)),
                         tuple(zip(*people)))

    def test_list_comprehensions(self):
        """List comprehensions are a concise way to transform lists.

        lst = [<operation> for <elt> in <iterable> [if <condition>]]

        <operation> : executed for each element in the iterable.
        <elt> : the current element being executed
        <iterable> : the iterable to run the list comprehension on
        <condition> : the condition to apply. If condition returns false,
                      the element is skipped.
        """

        squares = [x**2 for x in range(1, 4)]
        self.assertEqual([1, 4, 9], squares)

        evens = [x for x in range(1, 11) if x % 2 == 0]
        self.assertEqual([2, 4, 6, 8, 10], evens)

        squares = [(x, x ** 2) for x in [0, 1, 2, 3]]
        self.assertEqual([(0, 0), (1, 1), (2, 4), (3, 9)],
                         squares)
        # Note that the condition can only include an if statement.
        # If you want to use an else, you'll need to make the condition part of
        # the operation.

        # Here, we'll only compute squares for even numbers and default
        # odds to 0.
        squares_with_default = [x**2 if x % 2 == 0 else 0 for x in range(8)]
        self.assertEqual([0, 0, 4, 0, 16, 0, 36, 0],
                         squares_with_default)

        # You can have multiple for and if statements in the same list
        # comprehension.
        concatenated = [(x, y) for x in range(1, 3)
                        for y in range(1, 3) if x != y]
        self.assertListEqual([(1, 2), (2, 1)], concatenated)

        # Flatten a list using a listcomp
        vec = [[1, 2, 3], [4, 5, 6]]
        self.assertListEqual([1, 2, 3, 4, 5, 6], [
                             x for elt in vec for x in elt])

    def test_udacity_intro_to_python_data_structures(self):
        """This test shows a more involved example problem
           using iteration and list comprehensions."""

        nominated = {1931: ['Norman Taurog', 'Wesley Ruggles', 'Clarence Brown', 'Lewis Milestone', 'Josef Von Sternberg'],
                     1932: ['Frank Borzage', 'King Vidor', 'Josef Von Sternberg'],
                     1933: ['Frank Lloyd', 'Frank Capra', 'George Cukor'],
                     1934: ['Frank Capra', 'Victor Schertzinger', 'W. S. Van Dyke'],
                     1935: ['John Ford', 'Michael Curtiz', 'Henry Hathaway', 'Frank Lloyd'],
                     1936: ['Frank Capra', 'William Wyler', 'Robert Z. Leonard', 'Gregory La Cava', 'W. S. Van Dyke'],
                     1937: ['Leo McCarey', 'Sidney Franklin', 'William Dieterle', 'Gregory La Cava', 'William Wellman'],
                     1938: ['Frank Capra', 'Michael Curtiz', 'Norman Taurog', 'King Vidor', 'Michael Curtiz'],
                     1939: ['Sam Wood', 'Frank Capra', 'John Ford', 'William Wyler', 'Victor Fleming'],
                     1940: ['John Ford', 'Sam Wood', 'William Wyler', 'George Cukor', 'Alfred Hitchcock'],
                     1941: ['John Ford', 'Orson Welles', 'Alexander Hall', 'William Wyler', 'Howard Hawks'],
                     1942: ['Sam Wood', 'Mervyn LeRoy', 'John Farrow', 'Michael Curtiz', 'William Wyler'],
                     1943: ['Michael Curtiz', 'Ernst Lubitsch', 'Clarence Brown', 'George Stevens', 'Henry King'],
                     1944: ['Leo McCarey', 'Billy Wilder', 'Otto Preminger', 'Alfred Hitchcock', 'Henry King'],
                     1945: ['Billy Wilder', 'Leo McCarey', 'Clarence Brown', 'Jean Renoir', 'Alfred Hitchcock'],
                     1946: ['David Lean', 'Frank Capra', 'Robert Siodmak', 'Clarence Brown', 'William Wyler'],
                     1947: ['Elia Kazan', 'Henry Koster', 'Edward Dmytryk', 'George Cukor', 'David Lean'],
                     1948: ['John Huston', 'Laurence Olivier', 'Jean Negulesco', 'Fred Zinnemann', 'Anatole Litvak'],
                     1949: ['Joseph L. Mankiewicz', 'Robert Rossen', 'William A. Wellman', 'Carol Reed', 'William Wyler'],
                     1950: ['Joseph L. Mankiewicz', 'John Huston', 'George Cukor', 'Billy Wilder', 'Carol Reed'],
                     1951: ['George Stevens', 'John Huston', 'Vincente Minnelli', 'William Wyler', 'Elia Kazan'],
                     1952: ['John Ford', 'Joseph L. Mankiewicz', 'Cecil B. DeMille', 'Fred Zinnemann', 'John Huston'],
                     1953: ['Fred Zinnemann', 'Charles Walters', 'William Wyler', 'George Stevens', 'Billy Wilder'],
                     1954: ['Elia Kazan', 'George Seaton', 'William Wellman', 'Alfred Hitchcock', 'Billy Wilder'],
                     1955: ['Delbert Mann', 'John Sturges', 'Elia Kazan', 'Joshua Logan', 'David Lean'],
                     1956: ['George Stevens', 'Michael Anderson', 'William Wyler', 'Walter Lang', 'King Vidor'],
                     1957: ['David Lean', 'Mark Robson', 'Joshua Logan', 'Sidney Lumet', 'Billy Wilder'],
                     1958: ['Richard Brooks', 'Stanley Kramer', 'Robert Wise', 'Mark Robson', 'Vincente Minnelli'],
                     1959: ['George Stevens', 'Fred Zinnemann', 'Jack Clayton', 'Billy Wilder', 'William Wyler'],
                     1960: ['Billy Wilder', 'Jules Dassin', 'Alfred Hitchcock', 'Jack Cardiff', 'Fred Zinnemann'],
                     1961: ['J. Lee Thompson', 'Robert Rossen', 'Stanley Kramer', 'Federico Fellini', 'Robert Wise', 'Jerome Robbins'],
                     1962: ['David Lean', 'Frank Perry', 'Pietro Germi', 'Arthur Penn', 'Robert Mulligan'],
                     1963: ['Elia Kazan', 'Otto Preminger', 'Federico Fellini', 'Martin Ritt', 'Tony Richardson'],
                     1964: ['George Cukor', 'Peter Glenville', 'Stanley Kubrick', 'Robert Stevenson', 'Michael Cacoyannis'],
                     1965: ['William Wyler', 'John Schlesinger', 'David Lean', 'Hiroshi Teshigahara', 'Robert Wise'],
                     1966: ['Fred Zinnemann', 'Michelangelo Antonioni', 'Claude Lelouch', 'Richard Brooks', 'Mike Nichols'],
                     1967: ['Arthur Penn', 'Stanley Kramer', 'Richard Brooks', 'Norman Jewison', 'Mike Nichols'],
                     1968: ['Carol Reed', 'Gillo Pontecorvo', 'Anthony Harvey', 'Franco Zeffirelli', 'Stanley Kubrick'],
                     1969: ['John Schlesinger', 'Arthur Penn', 'George Roy Hill', 'Sydney Pollack', 'Costa-Gavras'],
                     1970: ['Franklin J. Schaffner', 'Federico Fellini', 'Arthur Hiller', 'Robert Altman', 'Ken Russell'],
                     1971: ['Stanley Kubrick', 'Norman Jewison', 'Peter Bogdanovich', 'John Schlesinger', 'William Friedkin'],
                     1972: ['Bob Fosse', 'John Boorman', 'Jan Troell', 'Francis Ford Coppola', 'Joseph L. Mankiewicz'],
                     1973: ['George Roy Hill', 'George Lucas', 'Ingmar Bergman', 'William Friedkin', 'Bernardo Bertolucci'],
                     1974: ['Francis Ford Coppola', 'Roman Polanski', 'Francois Truffaut', 'Bob Fosse', 'John Cassavetes'],
                     1975: ['Federico Fellini', 'Stanley Kubrick', 'Sidney Lumet', 'Robert Altman', 'Milos Forman'],
                     1976: ['Alan J. Pakula', 'Ingmar Bergman', 'Sidney Lumet', 'Lina Wertmuller', 'John G. Avildsen'],
                     1977: ['Steven Spielberg', 'Fred Zinnemann', 'George Lucas', 'Herbert Ross', 'Woody Allen'],
                     1978: ['Hal Ashby', 'Warren Beatty', 'Buck Henry', 'Woody Allen', 'Alan Parker', 'Michael Cimino'],
                     1979: ['Bob Fosse', 'Francis Coppola', 'Peter Yates', 'Edouard Molinaro', 'Robert Benton'],
                     1980: ['David Lynch', 'Martin Scorsese', 'Richard Rush', 'Roman Polanski', 'Robert Redford'],
                     1981: ['Louis Malle', 'Hugh Hudson', 'Mark Rydell', 'Steven Spielberg', 'Warren Beatty'],
                     1982: ['Wolfgang Petersen', 'Steven Spielberg', 'Sydney Pollack', 'Sidney Lumet', 'Richard Attenborough'],
                     1983: ['Peter Yates', 'Ingmar Bergman', 'Mike Nichols', 'Bruce Beresford', 'James L. Brooks'],
                     1984: ['Woody Allen', 'Roland Joffe', 'David Lean', 'Robert Benton', 'Milos Forman'],
                     1985: ['Hector Babenco', 'John Huston', 'Akira Kurosawa', 'Peter Weir', 'Sydney Pollack'],
                     1986: ['David Lynch', 'Woody Allen', 'Roland Joffe', 'James Ivory', 'Oliver Stone'],
                     1987: ['Bernardo Bertolucci', 'Adrian Lyne', 'John Boorman', 'Norman Jewison', 'Lasse Hallstrom'],
                     1988: ['Barry Levinson', 'Charles Crichton', 'Martin Scorsese', 'Alan Parker', 'Mike Nichols'],
                     1989: ['Woody Allen', 'Peter Weir', 'Kenneth Branagh', 'Jim Sheridan', 'Oliver Stone'],
                     1990: ['Francis Ford Coppola', 'Martin Scorsese', 'Stephen Frears', 'Barbet Schroeder', 'Kevin Costner'],
                     1991: ['John Singleton', 'Barry Levinson', 'Oliver Stone', 'Ridley Scott', 'Jonathan Demme'],
                     1992: ['Clint Eastwood', 'Neil Jordan', 'James Ivory', 'Robert Altman', 'Martin Brest'],
                     1993: ['Jim Sheridan', 'Jane Campion', 'James Ivory', 'Robert Altman', 'Steven Spielberg'],
                     1994: ['Woody Allen', 'Quentin Tarantino', 'Robert Redford', 'Krzysztof Kieslowski', 'Robert Zemeckis'],
                     1995: ['Chris Noonan', 'Tim Robbins', 'Mike Figgis', 'Michael Radford', 'Mel Gibson'],
                     1996: ['Anthony Minghella', 'Joel Coen', 'Milos Forman', 'Mike Leigh', 'Scott Hicks'],
                     1997: ['Peter Cattaneo', 'Gus Van Sant', 'Curtis Hanson', 'Atom Egoyan', 'James Cameron'],
                     1998: ['Roberto Benigni', 'John Madden', 'Terrence Malick', 'Peter Weir', 'Steven Spielberg'],
                     1999: ['Spike Jonze', 'Lasse Hallstrom', 'Michael Mann', 'M. Night Shyamalan', 'Sam Mendes'],
                     2000: ['Stephen Daldry', 'Ang Lee', 'Steven Soderbergh', 'Ridley Scott', 'Steven Soderbergh'],
                     2001: ['Ridley Scott', 'Robert Altman', 'Peter Jackson', 'David Lynch', 'Ron Howard'],
                     2002: ['Rob Marshall', 'Martin Scorsese', 'Stephen Daldry', 'Pedro Almodovar', 'Roman Polanski'],
                     2003: ['Fernando Meirelles', 'Sofia Coppola', 'Peter Weir', 'Clint Eastwood', 'Peter Jackson'],
                     2004: ['Martin Scorsese', 'Taylor Hackford', 'Alexander Payne', 'Mike Leigh', 'Clint Eastwood'],
                     2005: ['Ang Lee', 'Bennett Miller', 'Paul Haggis', 'George Clooney', 'Steven Spielberg'],
                     2006: ['Alejandro Gonzaalez Inarritu', 'Clint Eastwood', 'Stephen Frears', 'Paul Greengrass', 'Martin Scorsese'],
                     2007: ['Julian Schnabel', 'Jason Reitman', 'Tony Gilroy', 'Paul Thomas Anderson', 'Joel Coen', 'Ethan Coen'],
                     2008: ['David Fincher', 'Ron Howard', 'Gus Van Sant', 'Stephen Daldry', 'Danny Boyle'],
                     2009: ['James Cameron', 'Quentin Tarantino', 'Lee Daniels', 'Jason Reitman', 'Kathryn Bigelow'],
                     2010: ['Darren Aronofsky', 'David O. Russell', 'David Fincher', 'Ethan Coen', 'Joel Coen', 'Tom Hooper']}
        winners = {1931: ['Norman Taurog'],
                   1932: ['Frank Borzage'],
                   1933: ['Frank Lloyd'],
                   1934: ['Frank Capra'],
                   1935: ['John Ford'],
                   1936: ['Frank Capra'],
                   1937: ['Leo McCarey'],
                   1938: ['Frank Capra'],
                   1939: ['Victor Fleming'],
                   1940: ['John Ford'],
                   1941: ['John Ford'],
                   1942: ['William Wyler'],
                   1943: ['Michael Curtiz'],
                   1944: ['Leo McCarey'],
                   1945: ['Billy Wilder'],
                   1946: ['William Wyler'],
                   1947: ['Elia Kazan'],
                   1948: ['John Huston'],
                   1949: ['Joseph L. Mankiewicz'],
                   1950: ['Joseph L. Mankiewicz'],
                   1951: ['George Stevens'],
                   1952: ['John Ford'],
                   1953: ['Fred Zinnemann'],
                   1954: ['Elia Kazan'],
                   1955: ['Delbert Mann'],
                   1956: ['George Stevens'],
                   1957: ['David Lean'],
                   1958: ['Vincente Minnelli'],
                   1959: ['William Wyler'],
                   1960: ['Billy Wilder'],
                   1961: ['Jerome Robbins', 'Robert Wise'],
                   1962: ['David Lean'],
                   1963: ['Tony Richardson'],
                   1964: ['George Cukor'],
                   1965: ['Robert Wise'],
                   1966: ['Fred Zinnemann'],
                   1967: ['Mike Nichols'],
                   1968: ['Carol Reed'],
                   1969: ['John Schlesinger'],
                   1970: ['Franklin J. Schaffner'],
                   1971: ['William Friedkin'],
                   1972: ['Bob Fosse'],
                   1973: ['George Roy Hill'],
                   1974: ['Francis Ford Coppola'],
                   1975: ['Milos Forman'],
                   1976: ['John G. Avildsen'],
                   1977: ['Woody Allen'],
                   1978: ['Michael Cimino'],
                   1979: ['Robert Benton'],
                   1980: ['Robert Redford'],
                   1981: ['Warren Beatty'],
                   1982: ['Richard Attenborough'],
                   1983: ['James L. Brooks'],
                   1984: ['Milos Forman'],
                   1985: ['Sydney Pollack'],
                   1986: ['Oliver Stone'],
                   1987: ['Bernardo Bertolucci'],
                   1988: ['Barry Levinson'],
                   1989: ['Oliver Stone'],
                   1990: ['Kevin Costner'],
                   1991: ['Jonathan Demme'],
                   1992: ['Clint Eastwood'],
                   1993: ['Steven Spielberg'],
                   1994: ['Robert Zemeckis'],
                   1995: ['Mel Gibson'],
                   1996: ['Anthony Minghella'],
                   1997: ['James Cameron'],
                   1998: ['Steven Spielberg'],
                   1999: ['Sam Mendes'],
                   2000: ['Steven Soderbergh'],
                   2001: ['Ron Howard'],
                   2002: ['Roman Polanski'],
                   2003: ['Peter Jackson'],
                   2004: ['Clint Eastwood'],
                   2005: ['Ang Lee'],
                   2006: ['Martin Scorsese'],
                   2007: ['Ethan Coen', 'Joel Coen'],
                   2008: ['Danny Boyle'],
                   2009: ['Kathryn Bigelow'],
                   2010: ['Tom Hooper']}

        # 1A: Create dictionary with the count of Oscar nominations for each
        # director
        nom_count_dict = {}

        for nominated_lst in nominated.values():
            for director in nominated_lst:
                nom_count_dict[director] = nom_count_dict.get(director, 0) + 1

        # 1B: Create dictionary with the count of Oscar wins for each director
        win_count_dict = {}
        for winner_lst in winners.values():
            for winner in winner_lst:
                win_count_dict[winner] = win_count_dict.get(winner, 0) + 1

        # print("win_count_dict = {}".format(win_count_dict))

        # For Question 2: Please provide a list with the name(s) of the director(s) with
        # the most Oscar wins. The list can hold the names of multiple directors,
        # since there can be more than 1 director tied with the most Oscar wins.

        most_win_director = []
        max_wins = max(win_count_dict.values())

        most_win_director = [winner for winner,
                             count in win_count_dict.items() if count == max_wins]

        self.assertListEqual(["John Ford"], most_win_director)

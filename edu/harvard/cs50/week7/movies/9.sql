-- list the names people who starred in a movie released in 2004 ordered by birth year

select p.name from people as p join movies as m join stars as s on p.id = s.person_id and m.id = s.movie_id where m.year = 2004 order by p.birth asc;
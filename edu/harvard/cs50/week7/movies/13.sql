-- list the names of all people who starred in a movie in which Kevin Bacon also starred

with bacon_movies as(
    select distinct(m.id)
    from movies as m
        join people as p
        join stars as s on s.person_id = p.id and s.movie_id = m.id
      where p.name = 'Kevin Bacon' and birth = 1958  -- there are multiple 'Kevin Bacon's in the DB
)
select distinct(p.name) from movies as m join people as p join stars as s join bacon_movies as bm on s.person_id = p.id and s.movie_id = m.id and m.id = bm.id where p.name != 'Kevin Bacon';

-- list the titles of all movies in which both bradley cooper and jennifer lawrence starred

with lawrence_movies as(
    select distinct(m.id)
    from movies as m
        join people as p
        join stars as s on s.person_id = p.id and s.movie_id = m.id
      where p.name = 'Jennifer Lawrence'
)
select m.title from movies as m join people as p join stars as s join lawrence_movies as lm on s.person_id = p.id and s.movie_id = m.id where p.name = 'Bradley Cooper' and m.id = lm.id;

-- list the names of all people who starred in toy story

select p.name from people as p join movies as m join stars as s on p.id = s.person_id and m.id = s.movie_id where m.title like 'toy story';
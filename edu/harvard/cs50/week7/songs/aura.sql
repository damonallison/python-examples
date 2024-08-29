-- valence = positivity / negativity
-- high valence == happy - major keys and faster tempos
-- low valence == sad    - sad, subdued
--
--
-- energy = 0.65
-- danceability = 0.71
-- valence = 0.48

select avg(energy), avg(danceability), avg(valence) from songs;
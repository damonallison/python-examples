--
-- Write a SQL query that returns average energy of songs that are by Drake
--

SELECT AVG(songs.energy) FROM songs join artists on songs.artist_id = artists.id where artists.name = 'Drake';
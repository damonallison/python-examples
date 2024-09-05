-- Keep a log of any SQL queries you execute as you solve the mystery.

-- the theft took place on 7/28/2023 on Humphrey Street

-- When and where did the theft take place?
--
-- 10:15 AM
-- Humphrey Street Bakery
select * from crime_scene_reports where year = 2023 and month = 7 and day = 28;

-- What other evidence do we have from interviews?
--
-- I don't know the thief's name, but it was someone I recognized. Earlier this morning, before I arrived at Emma's bakery, I was walking by the ATM on Leggett Street and saw the thief there withdrawing some money.
-- Sometime within ten minutes of the theft, I saw the thief get into a car in the bakery parking lot and drive away. If you have security footage from the bakery parking lot, you might want to look for cars that left the parking lot in that time frame.
-- As the thief was leaving the bakery, they called someone who talked to them for less than a minute. In the call, I heard the thief say that they were planning to take the earliest flight out of Fiftyville tomorrow. The thief then asked the person on the other end of the phone to purchase the flight ticket.

-- Withdrew from ATM on Leggett Street (atm_transactions)
-- Within 10 minutes of 10:15 AM, security footage may show cars that left the parking lot (bakery_security_logs)
-- As he was leaving he called the accomplice (phone_calls)
-- They took the earliest flight out of Fiftyville on 7/29/2023 (flights)

select * from interviews where year = 2023 and month = 7 and day = 28;

-- Who withdrew from the ATM on Leggett Street?
select * from atm_transactions where year = 2023 and month = 7 and day = 28 and atm_location = 'Leggett Street' and transaction_type = 'withdraw';

-- What accounts are on the withdrawals?
select * from bank_accounts where account_number in (28500762,
28296815,
76054385,
49610011,
16153065,
25506511,
81061156,
26013199
);

-- Who owns those accounts?
--
-- Passports
-- ----------
-- 9878712108
-- 7049073643
-- 9586786673
-- 1988161715
-- 4408372428
-- 8496433585
-- 3592750733
-- 5773159633

-- License Plates
-- --------------
-- 30G67EN
-- L93JTIZ
-- 8X428L0
-- 1106N58
-- QX4YZN3
-- 4328GD8
-- 322W7JE
-- 94KL13X


-- Here are all the potential suspects
select * from people where id in (
686048,
514354,
458378,
395717,
396669,
467400,
449774,
438727
);

-- Which of these license_plates were at the bakery between 10:05 and 10:25?
-- 94KL13X
-- 4328GD8
-- L93JTIZ
-- 322W7JE

select * from bakery_security_logs where year = 2023 and month = 7 and day = 28 and hour = 10 and minute >= 5 and minute <= 25 and activity = 'exit' and license_plate in (
'30G67EN',
'L93JTIZ',
'8X428L0',
'1106N58',
'QX4YZN3',
'4328GD8',
'322W7JE',
'94KL13X'
);


-- Who owns these license plates?
select * from people where license_plate in (
'94KL13X',
'4328GD8',
'L93JTIZ',
'322W7JE'
);

-- Here are the new potential suspects
select * from people where id in (
396669,
467400,
514354,
686048);


-- Who made phone calls between 10:05 and 10:25?
select * from phone_calls pc join people p on pc.caller = p.phone_number where year = 2023 and month = 7 and day = 28 and p.id in (
396669,
467400,
514354,
686048);

-- Here are the new potential suspects
select * from people where id in (
514354,
686048
);


-- what is the earliest flight out from fiftyville on 7/29/2023
select *
from flights join airports on flights.origin_airport_id = airports.id
where year = 2023 and month = 7 and day = 29 and city = 'Fiftyville' order by hour, minute limit 1;

-- Who from within our potential people were on the earliest flight out?
--
-- flights.id = 36

select * from people join passengers on people.passport_number = passengers.passport_number join flights on passengers.flight_id = flights.id
where flight_id = 36
and people.id in (
514354,
686048
);

-- Here is the person who committed the crime: 686048 (Bruce)

-- What city did he escape to?: New York City

select * from flights join airports on flights.destination_airport_id = airports.id where flights.id = 36;

-- Who was the thief's accomplice who helped him escape?
-- Who did he call?
select * from phone_calls as pc join people as p on pc.caller = p.phone_number where pc.id in (233,236,245,285);

-- List of numbers he called
--
-- (375) 555-8161
-- (344) 555-9601
-- (022) 555-4052
-- (704) 555-5790



select * from people where phone_number in (
'(375) 555-8161',
'(344) 555-9601',
'(022) 555-4052',
'(704) 555-5790');

-- Potential accomplices
-- 315221
-- 652398
-- 864400 (no passport, can't fly
-- 985497

-- Who could have purchased the ticket?

select * from bank_accounts where person_id in (
315221,
652398,
864400,
985497);

-- His accomplice: 864400 (Robin)

select * from people where id = 864400;

-- Who was on the flight out?
-- Notice that Robin was not on the flight out (no passport) the crime scene mentioned a "single ticket" was purchased
select * from passengers join people on passengers.passport_number = people.passport_number where passengers.flight_id = 36 and people.id in (
315221,652398,864400, 985497);


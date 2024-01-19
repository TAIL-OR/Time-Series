with flights_ww as (
select strftime('%Y-%W', 
    substr(coalesce(chegada_real, chegada_prevista), 7, 4) || '-' || 
    substr(coalesce(chegada_real, chegada_prevista), 4, 2) || '-' || 
    substr(coalesce(chegada_real, chegada_prevista), 1, 2)) as week,
    count(*) as flights
from flights_data
where situação_voo = 'REALIZADO'
group by week
)

select 
case when cast(substr(week, 6, 2) as int) + 1 < 10 
    then substr(week, 1, 4) || '-0' || cast(cast(substr(week, 6, 2) as int) + 1 as text) 
    else substr(week, 1, 4) || '-' || cast(cast(substr(week, 6, 2) as int) + 1 as text) end as week, flights
from flights_ww

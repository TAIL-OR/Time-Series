with step1 as ( 
select *, strftime('%Y-%W', 
    substr(dataprimeirosintomas, 7, 4) || '-' || 
    substr(dataprimeirosintomas, 4, 2) || '-' || 
    substr(dataprimeirosintomas, 1, 2)) as week
from historical_data 
),
all_weeks as (
select distinct week from step1
),
ra_presence as (
select ra, week
from step1
where ra is not null and trim(ra) <> '' and trim(ra) <> 'Não Informado'
group by ra, week
),
weeks_with_ra as (
select aw.week, rp.ra
from all_weeks aw
cross join (select distinct ra from ra_presence) rp
)

select wwr.week, cast(substr(wwr.week, 6, 2) as int) as week_num,
wwr.ra, coalesce(count(s1.ra), 0) as total
from weeks_with_ra wwr
left join step1 s1 on s1.week = wwr.week and s1.ra = wwr.ra
group by wwr.week, wwr.ra
order by wwr.week asc, wwr.ra asc


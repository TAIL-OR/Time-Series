with step1 as (
select uf, ra, óbito, strftime('%Y-%m-%d', 
    substr(dataprimeirosintomas, 7, 4) || '-' || 
    substr(dataprimeirosintomas, 4, 2) || '-' || 
    substr(dataprimeirosintomas, 1, 2)) as data
from historical_data 
where dataprimeirosintomas <> '' or dataprimeirosintomas is not null
),

step2 as (
select data, ra, count(*) as case_cnt, 
    sum(case when lower(trim(óbito)) != 'não' then 1 else 0 end) as death_cnt
from step1
group by data, ra
),

step3 as (
select s2.*, min(
    case when s2.case_cnt >= 100 then s2.data end) over (partition by s2.ra) as date_100_cases
from step2 s2
)

select 
'brasilia' as country, lower(s3.ra) as province, s3.data as date, cast(julianday(s3.data) - julianday(s3.date_100_cases) as int) as day_since100,
s3.case_cnt, s3.death_cnt
from step3 s3
where s3.ra is not null and trim(ra) <> 'Não Informado' and trim(ra) <> ''
order by s3.data


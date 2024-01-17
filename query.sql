with step1 as ( 
select *, strftime('%Y-%W', 
    substr(dataprimeirosintomas, 7, 4) || '-' || 
    substr(dataprimeirosintomas, 4, 2) || '-' || 
    substr(dataprimeirosintomas, 1, 2)) as week
from historical_data 
)

select 
    week, substr(week, 6, 2) as week_num,
    ra,count(*) as total
from step1
where ra is not null and trim(ra) <> ''
group by week, ra
order by week asc

-- get percentual of nulls

-- select 
--     SUM(CASE WHEN ra IS NULL OR trim(ra) = '' THEN 1 ELSE 0 END) as dados_nulos_vazios,
--     (SUM(CASE WHEN ra IS NULL OR trim(ra) = '' THEN 1 ELSE 0 END) * 100.0) / COUNT(*) as percentual_nulos_vazios
-- from step1




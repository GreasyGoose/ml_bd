# HW02: Hive


## Блок 1. Развертывание локального Hive

1) Развернуть локальный Hive в любой конфигурации - 20 баллов
2) Подключиться к развернутому Hive с помощью любого инструмента: Hue, Python
Driver, Zeppelin, любая IDE итд (15 баллов за любой инструмент, максимум 30
баллов)
3) Сделать скриншоты поднятого Hive и подключений в выбранными вами
инструментах, добавить в репозиторий

-----------------------------------------
1) Hive

Hive-repository:

https://github.com/tech4242/docker-hadoop-hive-parquet


```
git clone https://github.com/tech4242/docker-hadoop-hive-parquet.git
cd docker-hadoop-hive-parquet
ls
nano docker-compose.yml
```

-> add port to resource manager
    ports:
      - 8088:8088

```docker-compose up```

-> wait util startup.
![alt text](01_containers_run.png)

-> After startup, in new terminal window
![alt text](02_hive_command_console.png)

```
cd docker-hadoop-hive-parquet
docker-compose exec hive-server bash
/opt/hive/bin/beeline -u jdbc:hive2://localhost:10000
```

```
show tables;
CREATE TABLE pokes (foo INT, bar STRING);
LOAD DATA LOCAL INPATH '/opt/hive/examples/files/kv1.txt' OVERWRITE INTO TABLE pokes;
select * from pokes limit 10;
```

```
Ctrl+c
exit
```
----------------------------------------------

2) Hue

![alt text](03_hue_loaded.png)

(If 
hue ProgrammingError: relation "desktop_settings" does not exist
not loaded, restart in docker app [see screenshot1])

in browser got to

localhost:8888

create login, psw

![alt text](04_hue_browser.png)


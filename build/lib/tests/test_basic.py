import json
import os

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from ddbtools.ddbtools import Connection, jslist, json_wrap


@pytest.fixture
def conn():
    def todf(text):
        lines = [line.strip().split() for line in text.split("\n")[1:-1]]
        header, data = lines[0], lines[1:]
        rows = []
        for line in data:
            rows.append(dict(zip(header, line)))
        return pd.DataFrame(rows)

    emp = """
    EMPNO ENAME JOB MGR HIREDATE SAL COMM DEPTNO
    7369 SMITH CLERK 7902 17-DEC-2005 800 '' 20
    7499 ALLEN SALESMAN 7698 20-FEB-2006 1600 300 30
    7521 WARD SALESMAN 7698 22-FEB-2006 1250 500 30
    7566 JONES MANAGER 7839 02-APR-2006 2975 '' 20
    7654 MARTIN SALESMAN 7698 28-SEP-2006 1250 1400 30
    7698 BLAKE MANAGER 7839 01-MAY-2006 2850 '' 30
    7782 CLARK MANAGER 7839 09-JUN-2006 2450 '' 10
    7788 SCOTT ANALYST 7566 09-DEC-2007 3000 '' 20
    7839 KING PRESIDENT '' 17-NOV-2006 5000 '' 10
    7844 TURNER SALESMAN 7698 08-SEP-2006 1500 0 30
    7876 ADAMS CLERK 7788 12-JAN-2008 1100 '' 20
    7900 JAMES CLERK 7698 03-DEC-2006 950 '' 30
    7902 FORD ANALYST 7566 03-DEC-2006 3000 '' 20
    7934 MILLER CLERK 7782 23-JAN-2007 1300 '' 10
    """

    dept = """
    DEPTNO DNAME LOC
    10 ACCOUNTING NEW_YORK
    20 RESEARCH DALLAS
    30 SALES CHICAGO
    40 OPERATIONS BOSTON
    """

    plant = """
    Plant	Date	MWh
    Boston	2019-01-02	564337
    Boston	2019-01-03	507405
    Boston	2019-01-04	528523
    Boston	2019-01-05	469538
    Boston	2019-01-06	474163
    Boston	2019-01-07	507213
    Boston	2019-01-08	613040
    Boston	2019-01-09	582588
    Boston	2019-01-10	499506
    Boston	2019-01-11	482014
    Boston	2019-01-12	486134
    Boston	2019-01-13	531518
    Worcester	2019-01-02	118860
    Worcester	2019-01-03	101977
    Worcester	2019-01-04	106054
    Worcester	2019-01-05	92182
    Worcester	2019-01-06	94492
    Worcester	2019-01-07	99932
    Worcester	2019-01-08	118854
    Worcester	2019-01-09	113506
    Worcester	2019-01-10	96644
    Worcester	2019-01-11	93806
    Worcester	2019-01-12	98963
    Worcester	2019-01-13	107170
    """

    dbpath = "testdb.duckdb"
    if os.path.exists(dbpath):
        os.remove(dbpath)
    con = Connection(dbpath)

    with con:
        con.register("emp_view", todf(emp))
        con.register("dept_view", todf(dept))
        con.register("plant_view", todf(plant))

        con.sql(
            """
        create table if not exists emp as
            select cast(empno as integer) as empno,
                ename, job,

                case when mgr = '' then NULL
                else try_cast(mgr as integer)
                end as mgr,

                strptime(hiredate, '%d-%b-%Y') as hiredate,

                cast(sal as integer) as sal,

                case when comm = '' then NULL
                else try_cast(comm as integer)
                end as comm,

                cast(deptno as integer) as deptno
            from emp_view
                    """
        )

        con.sql(
            """
        create table if not exists dept as
            select * replace(cast(deptno as integer) as deptno) from dept_view
                    """
        )
        con.sql(
            """
        create table if not exists plant as
            select Plant, cast(Date as timestamp) as Date, cast(MWh as int) as MWh
            from plant_view
            """
        )

    return con


def test_loading(conn):
    with conn:
        emp = conn.sql("select * from emp").df()
        assert len(emp) == 14
        dept = conn.sql("select * from dept").df()
        assert len(dept) == 4
        plant = conn.sql("select * from plant").df()
        assert len(plant) == 24


def test_itergroup(conn):
    with conn:
        for df, size in zip(
            conn.iter_group("emp", ["deptno", "job"]), [1, 1, 1, 2, 2, 1, 1, 1, 4]
        ):
            assert len(df) == size


def test_iterwindow(conn):
    with conn:
        for df, size in zip(
            conn.iter_window("emp", "hiredate", relativedelta(months=12)), [11, 2, 1]
        ):
            assert len(df) == size


def test_iterchunk(conn):
    with conn:
        for df, size in zip(conn.iter_chunk("emp", 5), [5, 5, 4]):
            assert len(df) == size


def test_push(conn):
    with conn:
        total_size = 0
        conn.sql("drop table if exists foo")
        for df in conn.iter_window(
            "emp", "hiredate", relativedelta(months=3), relativedelta(months=1)
        ):
            total_size += len(df)
            conn.push(df, "foo")
        assert total_size == len(conn.sql("select * from foo").df())


def test_js_maco_and_udf(conn):
    # js macro is case-sensitive for column identifiers
    # it's a bug, but not fixed yet
    @json_wrap
    def remove_top_n(mgrs: jslist, n: int) -> jslist:
        mgrs = sorted((x for x in mgrs if x), reverse=True)
        return mgrs[n:]

    with conn:
        conn.regfn(remove_top_n)
        df = conn.sql(
            """ 
            select
                deptno, 
                remove_top_n(js(mgr), 1) as mgr 
            from emp
            group by deptno
            order by deptno
        """
        ).df()

        for row, size in zip(df["mgr"], [1, 4, 5]):
            assert len(json.loads(row)) == size


def test_udf_interpolate(conn):
    @json_wrap
    def interpolate(before: jslist, val: int, after: jslist) -> float:
        def first_val(xs):
            if not xs:
                return None, None
            return next((x, i) for i, x in enumerate(xs, 1) if x)

        if not val:
            left, left_pos = first_val(reversed(before))
            right, right_pos = first_val(after)
            if left and right:
                return left + (right - left) * left_pos / (left_pos + right_pos)
        return val

    with conn:
        # delete some rows
        conn.sql(
            """
            delete from plant 
            where 
                (Date = DATE '2019-01-07' and plant = 'Boston') or  
                (Date = DATE '2019-01-08' and plant = 'Boston') or  
                (Date = DATE '2019-01-11' and plant = 'Boston') or  
                (Date = DATE '2019-01-12' and plant = 'Worcester')
                 """
        )

        # fillin the missing dates
        conn.sql(
            """
            create table plant1 as
            with refdates as (
                select date 
                from generate_series(DATE '2019-01-02', DATE '2019-01-13', interval '1 day') as t(date)
            ),
                refdates_plant as (
                 select b.plant, a.date
                 from refdates as a
                 cross join (select distinct plant from plant) as b 
            )
            select a.plant, a.date, b.MWh 
            from refdates_plant as a
                left join plant as b
                on a.date = b.date and a.plant = b.plant
            order by a.plant, a.date
                 """
        )

        conn.regfn(interpolate)
        df = conn.sql(
            """
            with roll as (
                select
                    Plant, Date,
                    list(MWh) over before as mwh_before, mwh,
                    list(MWh) over after as mwh_after
                from plant1
                    window before as (
                        partition by Plant order by Date ASC
                        range between interval 3 days preceding
                                and   interval 1 days preceding
                    ),
                           after as (
                        partition by Plant order by Date ASC
                        range between interval 1 days following
                                and   interval 3 days following
                           )
            )
            select 
                Plant, Date, 
                mwh_before, mwh, mwh_after,
                interpolate(to_json(mwh_before), mwh, to_json(mwh_after)) as val
            from roll
            where val is not null
        """
        ).df()

        assert len(df) == 24

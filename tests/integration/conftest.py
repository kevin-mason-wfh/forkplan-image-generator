"""Real-Postgres tests. Set TEST_DATABASE_URL (a throwaway database you may create schemas in) to run them; otherwise skipped."""
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


@pytest.fixture(autouse=True)
def forbid_real_network():
    """Overrides the suite-wide socket guard: these tests talk to a real database on purpose."""


@pytest.fixture
def engine():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to run the Postgres integration tests")
    import psycopg

    schema = f"it_{uuid.uuid4().hex[:10]}"
    plain_url = make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(plain_url, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
        with psycopg.connect(plain_url, autocommit=True, options=f"-c search_path={schema}") as conn:
            conn.execute("""
                CREATE TABLE catalog_recipes (id uuid PRIMARY KEY, name text NOT NULL, category text, instructions text,
                                              image_key text, status text NOT NULL DEFAULT 'PUBLISHED');
                CREATE TABLE catalog_ingredients (id uuid PRIMARY KEY, name text NOT NULL);
                CREATE TABLE catalog_recipe_ingredients (catalog_recipe_id uuid REFERENCES catalog_recipes(id),
                    catalog_ingredient_id uuid REFERENCES catalog_ingredients(id), quantity numeric, unit text);""")
            for migration in sorted(MIGRATIONS.glob("*.sql")):
                conn.execute(migration.read_text())
    engine = create_engine(make_url(url).set(drivername="postgresql+psycopg"),
                           connect_args={"options": f"-c search_path={schema}"})
    yield engine
    engine.dispose()
    with psycopg.connect(plain_url, autocommit=True) as admin:
        admin.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.fixture
def recipe_ids(engine):
    ids = [uuid.uuid4() for _ in range(3)]
    with engine.begin() as conn:
        for number, recipe_id in enumerate(ids):
            conn.execute(text("INSERT INTO catalog_recipes (id, name, instructions) VALUES (:i, :n, 'Cook it.')"),
                         {"i": recipe_id, "n": f"Dish {number}"})
    return ids

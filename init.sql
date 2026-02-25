CREATE TABLE IF NOT EXISTS document (
    id          SERIAL PRIMARY KEY,
    images      TEXT NOT NULL,
    latex       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_document_updated
    BEFORE UPDATE ON document
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

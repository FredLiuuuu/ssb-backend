ALTER TABLE records ADD COLUMN IF NOT EXISTS project_id VARCHAR;
ALTER TABLE records ADD COLUMN IF NOT EXISTS source VARCHAR;

UPDATE records SET project_id = 'default' WHERE project_id IS NULL;
UPDATE records SET source = 'chat' WHERE source IS NULL;

CREATE INDEX IF NOT EXISTS idx_records_project_id ON records(project_id);
CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);

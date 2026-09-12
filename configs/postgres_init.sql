-- ============================================================
-- configs/postgres_init.sql
-- 云诊医疗智能问诊系统 —— PostgreSQL 数据库初始化脚本
--
-- 表结构说明：
--   users             用户表（与 memory/long_memory.py 一致）
--   medical_records   病历表（与 memory/long_memory.py 一致）
--   consultations     问诊记录表（多轮对话明细）
--
-- 执行方式：
--   psql -U postgres -d cloud_diagnosis -f configs/postgres_init.sql
-- ============================================================


-- ============================================================
-- 1. 创建数据库（如不存在则创建）
-- ============================================================

-- SELECT 'CREATE DATABASE cloud_diagnosis'
-- WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'cloud_diagnosis')\gexec

-- \c cloud_diagnosis;


-- ============================================================
-- 2. 用户表 users
--    与 memory/long_memory.py 中的 ORM 模型完全一致
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    user_id         VARCHAR(64)   PRIMARY KEY,
    phone           VARCHAR(20)   UNIQUE NOT NULL,
    name            VARCHAR(64)   DEFAULT '',
    gender          VARCHAR(10)   DEFAULT '',
    birth_date      VARCHAR(20)   DEFAULT '',
    allergies       JSONB         DEFAULT '[]'::jsonb,
    chronic_diseases JSONB        DEFAULT '[]'::jsonb,
    created_at      TIMESTAMP     DEFAULT NOW(),
    updated_at      TIMESTAMP     DEFAULT NOW()
);

COMMENT ON TABLE  users IS '用户信息表';
COMMENT ON COLUMN users.user_id IS '用户唯一ID（UUID）';
COMMENT ON COLUMN users.phone IS '手机号（登录凭证）';
COMMENT ON COLUMN users.name IS '用户姓名';
COMMENT ON COLUMN users.gender IS '性别：男/女/未知';
COMMENT ON COLUMN users.birth_date IS '出生日期 YYYY-MM-DD';
COMMENT ON COLUMN users.allergies IS '过敏史 JSON 数组';
COMMENT ON COLUMN users.chronic_diseases IS '慢性病 JSON 数组';


-- ============================================================
-- 3. 病历表 medical_records
--    与 memory/long_memory.py 中的 ORM 模型完全一致
-- ============================================================

CREATE TABLE IF NOT EXISTS medical_records (
    record_id           VARCHAR(64)   PRIMARY KEY,
    user_id             VARCHAR(64)   NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id          VARCHAR(64)   NOT NULL,
    chief_complaint     TEXT          DEFAULT '',
    symptoms            JSONB         DEFAULT '[]'::jsonb,
    disease_candidates  JSONB         DEFAULT '[]'::jsonb,
    final_diagnosis     TEXT          DEFAULT '',
    prescriptions       JSONB         DEFAULT '[]'::jsonb,
    report_images       JSONB         DEFAULT '[]'::jsonb,
    report_analysis     TEXT          DEFAULT '',
    doctor_notes        TEXT          DEFAULT '',
    created_at          TIMESTAMP     DEFAULT NOW(),
    updated_at          TIMESTAMP     DEFAULT NOW()
);

COMMENT ON TABLE  medical_records IS '病历记录表';
COMMENT ON COLUMN medical_records.record_id IS '病历唯一ID（UUID）';
COMMENT ON COLUMN medical_records.user_id IS '关联用户ID';
COMMENT ON COLUMN medical_records.session_id IS '问诊会话ID';
COMMENT ON COLUMN medical_records.chief_complaint IS '主诉';
COMMENT ON COLUMN medical_records.symptoms IS '标准化症状列表 JSON';
COMMENT ON COLUMN medical_records.disease_candidates IS '候选疾病列表 JSON';
COMMENT ON COLUMN medical_records.final_diagnosis IS '最终诊断';
COMMENT ON COLUMN medical_records.prescriptions IS '处方列表 JSON';
COMMENT ON COLUMN medical_records.report_images IS '报告图片路径列表 JSON';
COMMENT ON COLUMN medical_records.report_analysis IS '报告解读结果';
COMMENT ON COLUMN medical_records.doctor_notes IS '医生备注';


-- ============================================================
-- 4. 问诊记录表 consultations（多轮对话明细）
--    记录每次问诊的完整对话过程，用于回溯和审计
-- ============================================================

CREATE TABLE IF NOT EXISTS consultations (
    consultation_id     VARCHAR(64)   PRIMARY KEY,
    user_id             VARCHAR(64)   NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id          VARCHAR(64)   NOT NULL,
    record_id           VARCHAR(64)   REFERENCES medical_records(record_id) ON DELETE SET NULL,
    intent              VARCHAR(32)   DEFAULT '',
    intent_confidence   DECIMAL(5,4)  DEFAULT 0.0000,
    inquiry_rounds      INTEGER       DEFAULT 0,
    max_rounds          INTEGER       DEFAULT 10,
    converged           BOOLEAN       DEFAULT FALSE,
    final_diagnosis     TEXT          DEFAULT '',
    diagnosis_confidence DECIMAL(5,4) DEFAULT 0.0000,
    standardized_symptoms JSONB       DEFAULT '[]'::jsonb,
    disease_candidates  JSONB         DEFAULT '[]'::jsonb,
    conversation_log    JSONB         DEFAULT '[]'::jsonb,
    has_image           BOOLEAN       DEFAULT FALSE,
    image_paths         JSONB         DEFAULT '[]'::jsonb,
    image_analysis      TEXT          DEFAULT '',
    started_at          TIMESTAMP     DEFAULT NOW(),
    ended_at            TIMESTAMP,
    duration_seconds    INTEGER,
    created_at          TIMESTAMP     DEFAULT NOW(),
    updated_at          TIMESTAMP     DEFAULT NOW()
);

COMMENT ON TABLE  consultations IS '问诊记录表（多轮对话明细）';
COMMENT ON COLUMN consultations.consultation_id IS '问诊记录唯一ID（UUID）';
COMMENT ON COLUMN consultations.user_id IS '关联用户ID';
COMMENT ON COLUMN consultations.session_id IS '会话ID';
COMMENT ON COLUMN consultations.record_id IS '关联病历ID';
COMMENT ON COLUMN consultations.intent IS '意图分类结果';
COMMENT ON COLUMN consultations.intent_confidence IS '意图分类置信度';
COMMENT ON COLUMN consultations.inquiry_rounds IS '实际追问轮数';
COMMENT ON COLUMN consultations.max_rounds IS '最大允许追问轮数';
COMMENT ON COLUMN consultations.converged IS '是否诊断收敛';
COMMENT ON COLUMN consultations.final_diagnosis IS '最终诊断结果';
COMMENT ON COLUMN consultations.diagnosis_confidence IS '诊断置信度';
COMMENT ON COLUMN consultations.standardized_symptoms IS '标准化症状列表 JSON';
COMMENT ON COLUMN consultations.disease_candidates IS '候选疾病列表 JSON';
COMMENT ON COLUMN consultations.conversation_log IS '完整对话日志 JSON [{role, content, timestamp}]';
COMMENT ON COLUMN consultations.has_image IS '是否包含图片';
COMMENT ON COLUMN consultations.image_paths IS '图片路径列表 JSON';
COMMENT ON COLUMN consultations.image_analysis IS '图片分析结果';
COMMENT ON COLUMN consultations.started_at IS '问诊开始时间';
COMMENT ON COLUMN consultations.ended_at IS '问诊结束时间';
COMMENT ON COLUMN consultations.duration_seconds IS '问诊持续秒数';


-- ============================================================
-- 5. 索引创建
-- ============================================================

-- 用户表索引
CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);

-- 病历表索引
CREATE INDEX IF NOT EXISTS idx_records_user_created
    ON medical_records(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_records_user_diagnosis
    ON medical_records(user_id, final_diagnosis)
    WHERE final_diagnosis != '';

CREATE INDEX IF NOT EXISTS idx_records_session
    ON medical_records(session_id);

-- 问诊记录表索引
CREATE INDEX IF NOT EXISTS idx_consultations_user_created
    ON consultations(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_consultations_session
    ON consultations(session_id);

CREATE INDEX IF NOT EXISTS idx_consultations_record
    ON consultations(record_id);

CREATE INDEX IF NOT EXISTS idx_consultations_diagnosis
    ON consultations(final_diagnosis)
    WHERE final_diagnosis != '';


-- ============================================================
-- 6. 自动更新 updated_at 触发器
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为各表绑定触发器
DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_records_updated_at ON medical_records;
CREATE TRIGGER trg_records_updated_at
    BEFORE UPDATE ON medical_records
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_consultations_updated_at ON consultations;
CREATE TRIGGER trg_consultations_updated_at
    BEFORE UPDATE ON consultations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- 7. 示例数据（可选）
-- ============================================================

-- 插入测试用户
INSERT INTO users (user_id, phone, name, gender, birth_date, allergies, chronic_diseases)
VALUES
    ('test_user_001', '13800138000', '张三', '男', '1990-01-15', '["青霉素"]'::jsonb, '[]'::jsonb),
    ('test_user_002', '13900139000', '李四', '女', '1985-06-20', '[]'::jsonb, '["高血压"]'::jsonb)
ON CONFLICT (user_id) DO NOTHING;

-- 插入测试病历
INSERT INTO medical_records (record_id, user_id, session_id, chief_complaint, symptoms, final_diagnosis)
VALUES
    ('rec_001', 'test_user_001', 'sess_001', '头痛发热三天',
     '[{"standard_name":"头痛","confidence":0.9},{"standard_name":"发热","confidence":0.85}]'::jsonb,
     '上呼吸道感染'),
    ('rec_002', 'test_user_002', 'sess_002', '腹痛腹泻两天',
     '[{"standard_name":"腹痛","confidence":0.88},{"standard_name":"腹泻","confidence":0.92}]'::jsonb,
     '急性胃肠炎')
ON CONFLICT (record_id) DO NOTHING;

-- 插入测试问诊记录
INSERT INTO consultations (consultation_id, user_id, session_id, record_id, intent, inquiry_rounds, converged, final_diagnosis)
VALUES
    ('con_001', 'test_user_001', 'sess_001', 'rec_001', 'consultation', 3, TRUE, '上呼吸道感染'),
    ('con_002', 'test_user_002', 'sess_002', 'rec_002', 'consultation', 2, TRUE, '急性胃肠炎')
ON CONFLICT (consultation_id) DO NOTHING;


-- ============================================================
-- 8. 验证查询（可选执行）
-- ============================================================

-- 查看所有表
-- SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;

-- 查看表结构
-- \d users
-- \d medical_records
-- \d consultations

-- 查看示例数据
-- SELECT * FROM users;
-- SELECT * FROM medical_records;
-- SELECT * FROM consultations;

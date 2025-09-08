-- Initialize Wellix database with required extensions and initial data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for password hashing (if needed)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create initial admin user (optional)
-- This would be handled by the application in production
-- INSERT INTO users (id, email, hashed_password, first_name, last_name, is_active, is_premium, created_at)
-- VALUES (
--     uuid_generate_v4(),
--     'admin@wellix.com',
--     '$2b$12$example_hashed_password',
--     'Admin',
--     'User',
--     true,
--     true,
--     NOW()
-- );

-- Create indexes for better performance (these will also be in migrations)
-- CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
-- CREATE INDEX IF NOT EXISTS idx_food_analyses_user_id ON food_analyses(user_id);
-- CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
-- CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);

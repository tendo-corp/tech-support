-- Who gets reports in a private chat; filled when a REPORT_TO user writes to the bot.
create table if not exists public.recipients (
  username   text primary key,
  chat_id    bigint not null,
  created_at timestamptz not null default now()
);
alter table public.recipients enable row level security;

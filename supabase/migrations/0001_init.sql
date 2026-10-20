-- Chats the bot is in, and which contract applies to each.
create table if not exists public.chats (
  chat_id     bigint primary key,
  title       text,
  contract_id text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Every message from customer chats; customer messages carry JEV's verdict.
create table if not exists public.messages (
  id                  bigint generated always as identity primary key,
  chat_id             bigint not null,
  message_id          bigint not null,
  thread_id           bigint,
  user_id             bigint,
  username            text,
  author              text,
  author_side         text not null check (author_side in ('client', 'team')),
  text                text not null default '',
  reply_to_message_id bigint,
  reply_to_text       text,
  has_photo           boolean not null default false,
  sent_at             timestamptz,
  edited              boolean not null default false,

  category            text check (category in ('bug', 'feature', 'question', 'discussion', 'noise')),
  is_task             boolean,
  in_scope            boolean,           -- true: в рамках контракта, false: выход за рамки
  contract_item       text,              -- ClickUp task id
  bug_probability     smallint,
  complexity          text,
  urgency             text,
  estimate_min_h      numeric,
  estimate_max_h      numeric,
  priority            text,
  needs_clarification boolean,
  answers             jsonb,             -- full questionnaire
  reasoning           text,
  model               text,              -- model JEV's router picked
  cost_usd            numeric,
  latency_s           numeric,
  error               text,

  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now(),
  unique (chat_id, message_id)
);

create index if not exists messages_chat_sent_idx on public.messages (chat_id, sent_at desc);
create index if not exists messages_open_tasks_idx on public.messages (category, in_scope) where is_task;

create or replace function public.touch_updated_at() returns trigger
language plpgsql set search_path = '' as $$
begin
  new.updated_at := now();
  return new;
end $$;

drop trigger if exists chats_touch on public.chats;
create trigger chats_touch before update on public.chats for each row execute function public.touch_updated_at();
drop trigger if exists messages_touch on public.messages;
create trigger messages_touch before update on public.messages for each row execute function public.touch_updated_at();

-- Only the service role (the bot) touches these tables; RLS with no policies blocks anon/authenticated keys.
alter table public.chats enable row level security;
alter table public.messages enable row level security;

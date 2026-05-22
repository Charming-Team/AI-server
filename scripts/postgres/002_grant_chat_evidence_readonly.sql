begin;

do $$
begin
    if not exists (
        select 1
        from pg_roles
        where rolname = 'smap_chat_reader'
    ) then
        create role smap_chat_reader login;
    end if;
end
$$;

do $$
begin
    execute format(
        'grant connect on database %I to smap_chat_reader',
        current_database()
    );
end
$$;

alter role smap_chat_reader set search_path = chat_evidence;
alter role smap_chat_reader set default_transaction_read_only = on;

revoke all privileges on schema public from smap_chat_reader;
revoke all privileges on all tables in schema public from smap_chat_reader;
revoke all privileges on all sequences in schema public from smap_chat_reader;
revoke all privileges on all functions in schema public from smap_chat_reader;

revoke all privileges on schema chat_evidence from smap_chat_reader;
revoke all privileges on all tables in schema chat_evidence from smap_chat_reader;

grant usage on schema chat_evidence to smap_chat_reader;
grant select on all tables in schema chat_evidence to smap_chat_reader;

alter default privileges in schema chat_evidence
    grant select on tables to smap_chat_reader;

comment on role smap_chat_reader is
    'FastAPI chatbot read-only role. Password must be managed by external secret.';

commit;

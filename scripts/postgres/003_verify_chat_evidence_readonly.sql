with chat_evidence_views as (
    select
        table_schema,
        table_name,
        format('%I.%I', table_schema, table_name) as qualified_name
    from information_schema.views
    where table_schema = 'chat_evidence'
),
view_permissions as (
    select
        table_schema,
        table_name,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'SELECT'
        ) as can_select,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'INSERT'
        ) as can_insert,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'UPDATE'
        ) as can_update,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'DELETE'
        ) as can_delete,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'TRUNCATE'
        ) as can_truncate
    from chat_evidence_views
)
select
    'chat_evidence_view_privilege' as check_name,
    table_schema,
    table_name,
    can_select,
    can_insert,
    can_update,
    can_delete,
    can_truncate
from view_permissions
order by table_name;

select
    'granted_privilege' as check_name,
    privilege_type,
    table_schema,
    table_name
from information_schema.role_table_grants
where grantee = 'smap_chat_reader'
order by table_schema, table_name, privilege_type;

with public_tables as (
    select
        table_schema,
        table_name,
        format('%I.%I', table_schema, table_name) as qualified_name
    from information_schema.tables
    where table_schema = 'public'
      and table_type = 'BASE TABLE'
),
public_table_permissions as (
    select
        table_schema,
        table_name,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'SELECT'
        ) as can_select,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'INSERT'
        ) as can_insert,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'UPDATE'
        ) as can_update,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'DELETE'
        ) as can_delete,
        has_table_privilege(
            'smap_chat_reader',
            qualified_name,
            'TRUNCATE'
        ) as can_truncate
    from public_tables
)
select
    'public_base_table_unexpected_privilege' as check_name,
    table_schema,
    table_name,
    can_select,
    can_insert,
    can_update,
    can_delete,
    can_truncate
from public_table_permissions
where can_select
   or can_insert
   or can_update
   or can_delete
   or can_truncate
order by table_name;

begin;

drop view if exists chat_evidence.chat_report_metadata_evidence_view;
drop view if exists chat_evidence.chat_work_priority_evidence_view;
drop view if exists chat_evidence.chat_urgent_order_impact_evidence_view;
drop view if exists chat_evidence.chat_line_bottleneck_evidence_view;
drop view if exists chat_evidence.chat_production_plan_evidence_view;
drop view if exists chat_evidence.chat_delivery_risk_evidence_view;
drop view if exists chat_evidence.chat_material_shortage_evidence_view;

commit;

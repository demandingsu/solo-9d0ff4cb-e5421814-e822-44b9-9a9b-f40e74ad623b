from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from training_portal import main as app_main
from training_portal.service import Portal


@pytest.fixture()
def portal(tmp_path):
    instance = Portal(tmp_path / "data")
    instance.start_workers()
    yield instance
    instance.shutdown()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    instance = Portal(tmp_path / "data")
    monkeypatch.setattr(app_main, "portal", instance)
    with TestClient(app_main.app) as test_client:
        yield test_client, instance
    instance.shutdown()


CSV_SPRING = """batch,respondent_id,text
B1,R1,春天第一段访谈正文
B1,R2,春天第二段访谈正文
B1,R3,春天第三段访谈正文
B2,R4,另一个批次的正文
"""


def upload(instance: Portal, csv_text: str, name: str = "u.csv") -> dict:
    return instance.ingest_upload(name, csv_text.encode("utf-8"))


def wait_for(instance: Portal, job_id: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for job in instance.jobs():
            if job["id"] == job_id:
                if job["status"] in ("completed", "blocked", "failed"):
                    return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not settle")


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def test_upload_parses_headers_and_reports_counts(portal):
    result = upload(portal, CSV_SPRING)
    assert result["accepted"] == 4
    assert result["batches"] == ["B1", "B2"]
    assert portal.stats()["utterances"] == 4


def test_upload_accepts_chinese_headers(portal):
    csv_text = "访谈批次,受访者编号,文本\nB1,R1,中文表头也支持\n"
    assert upload(portal, csv_text)["accepted"] == 1


def test_upload_rejects_missing_column(portal):
    from training_portal.service import ValidationError

    with pytest.raises(ValidationError):
        upload(portal, "batch,text\nB1,缺一列\n")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def test_generation_produces_csv_for_selected_batch(portal):
    upload(portal, CSV_SPRING)
    video = portal.create_job("B1")
    job = wait_for(portal, video)
    assert job["status"] == "completed"

    export = portal.exports()[0]
    handle, name, error = portal.open_download(export["id"])
    assert error is None and handle is not None
    content = handle.read().decode("utf-8")
    handle.close()
    assert "春天第一段访谈正文" in content
    assert "另一个批次的正文" not in content
    assert content.splitlines()[0] == "batch,respondent_id,text"


def test_generation_of_empty_batch_is_rejected(portal):
    from training_portal.service import ValidationError

    with pytest.raises(ValidationError):
        portal.create_job("does-not-exist")


# ---------------------------------------------------------------------------
# revocation
# ---------------------------------------------------------------------------
def test_revocation_deletes_rows_and_locks_existing_exports(portal):
    upload(portal, CSV_SPRING)
    job_id = portal.create_job("B1")
    wait_for(portal, job_id)
    export_id = portal.exports()[0]["id"]

    result = portal.revoke("R2")
    assert result["deleted_rows"] == 1
    assert result["affected_exports"] == 1

    handle, _name, error = portal.open_download(export_id)
    assert handle is None
    assert "撤回" in error
    assert not [e for e in portal.exports() if e["id"] == export_id and e["status"] == "ready"]


def test_revocation_keeps_unaffected_files_downloadable(portal):
    upload(portal, CSV_SPRING)
    job_b1 = portal.create_job("B1")
    job_b2 = portal.create_job("B2")
    wait_for(portal, job_b1)
    wait_for(portal, job_b2)
    exports = {e["batch"]: e for e in portal.exports()}

    portal.revoke("R1")

    handle, _name, error = portal.open_download(exports["B2"]["id"])
    assert error is None and handle is not None
    handle.close()


def test_revocation_blocks_later_upload_of_same_id(portal):
    upload(portal, CSV_SPRING)
    portal.revoke("R1")

    result = upload(portal, "batch,respondent_id,text\nB9,R1,重新上传同号文本\n")
    assert result["accepted"] == 0
    assert result["skipped_revoked"] == 1
    assert "B9" not in [b["name"] for b in portal.batches()]


def test_revocation_blocks_original_text_under_new_id(portal):
    upload(portal, CSV_SPRING)
    portal.revoke("R1")

    csv_text = "batch,respondent_id,text\nB9,R1-ALT,春天第一段访谈正文\n"
    result = upload(portal, csv_text)
    assert result["accepted"] == 0
    assert result["skipped_revoked"] == 1


def test_revocation_is_persistent_across_restart(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    upload(first, CSV_SPRING)
    first.revoke("R2")
    first.shutdown()

    second = Portal(data_dir)
    try:
        handle, _name, error = second.open_download(1) if second.exports() else (None, "", "")
        assert second.stats()["revocations"] == 1
        result = upload(second, "batch,respondent_id,text\nB1,R2,换新文本再试\n")
        assert result["skipped_revoked"] == 1
        assert handle is None
    finally:
        second.shutdown()


@pytest.mark.parametrize("respondent", ["R1", "R2", "R3", "R4"])
def test_revocation_of_any_respondent_leaves_consistent_state(portal, respondent):
    upload(portal, CSV_SPRING)
    job = portal.create_job("B1")
    wait_for(portal, job)
    portal.revoke(respondent)

    in_batch = respondent in {"R1", "R2", "R3"}
    for export in portal.exports():
        if export["status"] == "ready":
            handle, _name, error = portal.open_download(export["id"])
            assert error is None
            handle.close()

    export_id = portal.exports()[0]["id"]
    handle, _n, error = portal.open_download(export_id)
    if in_batch:
        assert handle is None and "撤回" in error
    else:
        # A respondent from another batch must not take this file down.
        assert handle is not None and error is None
        handle.close()


# ---------------------------------------------------------------------------
# mid-generation revocation
# ---------------------------------------------------------------------------
def test_revocation_during_generation_prevents_publication(client):
    test_client, instance = client
    repair = "\n".join(f"B1,R{i},第{i}段较长的访谈正文内容用于演示生成过程中的撤回" for i in range(200))
    upload(instance, "batch,respondent_id,text\n" + repair)
    instance.db.set_settings({"pause_generation": "1"})

    r = test_client.post("/generate", data={"batch": "B1"})
    assert r.status_code == 200

    deadline = time.monotonic() + 5
    paused = None
    while time.monotonic() < deadline:
        paused = next((j for j in instance.jobs() if j["status"] == "paused"), None)
        if paused:
            break
        time.sleep(0.05)
    assert paused is not None, "job never reached the pause gate"

    r = test_client.post("/revoke", data={"respondent_id": "R37"})
    assert r.status_code == 200

    instance.db.set_settings({"pause_generation": "0"})
    deadline = time.monotonic() + 5
    settled = None
    while time.monotonic() < deadline:
        settled = instance.jobs()[0]
        if settled["status"] in ("blocked", "completed", "failed"):
            break
        time.sleep(0.05)
    assert settled["status"] == "blocked"

    assert instance.exports() == []
    assert "R37" in r.text or "撤回" in r.text


# ---------------------------------------------------------------------------
# delete failures, pending cleanup, restarts
# ---------------------------------------------------------------------------
def test_delete_failure_leaves_pending_and_blocks_download(portal):
    upload(portal, CSV_SPRING)
    job = portal.create_job("B1")
    wait_for(portal, job)
    export_id = portal.exports()[0]["id"]

    portal.db.set_settings({"delete_failures_remaining": 2})
    portal.revoke("R2")

    deadline = time.monotonic() + 5
    export = None
    while time.monotonic() < deadline:
        export = next(e for e in portal.exports() if e["id"] == export_id)
        if export["status"] == "pending_delete" and export["delete_attempts"] >= 1:
            break
        time.sleep(0.05)
    assert export["status"] == "pending_delete"
    assert export["last_delete_error"]
    assert export["delete_attempts"] >= 1

    handle, _name, error = portal.open_download(export_id)
    assert handle is None and error

    # The file still exists on disk but is not reachable under a download name
    # until the cleanup budget is exhausted.
    path = portal.export_dir / export["stored_name"]
    assert path.exists()

    portal.db.set_settings({"delete_failures_remaining": 0})
    portal.retry_cleanup_now()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        export = next(e for e in portal.exports() if e["id"] == export_id)
        if export["status"] == "deleted":
            break
        time.sleep(0.05)
    assert export["status"] == "deleted"
    assert not path.exists()


def test_restart_continues_pending_cleanup_and_never_reopens(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    first.start_workers()
    try:
        upload(first, CSV_SPRING)
        job = first.create_job("B1")
        wait_for(first, job)
        export_id = first.exports()[0]["id"]
        first.db.set_settings({"delete_always_fail": "1"})
        first.revoke("R1")
        time.sleep(0.3)
        pending = next(e for e in first.exports() if e["id"] == export_id)
        assert pending["status"] == "pending_delete"
        assert (first.export_dir / pending["stored_name"]).exists()
        handle, _n, error = first.open_download(export_id)
        assert handle is None
    finally:
        first.shutdown()

    second = Portal(data_dir)
    second.start_workers()
    try:
        # The switch is still on after restart: cleanup retries and still fails,
        # and the file remains non-downloadable.
        time.sleep(0.4)
        row = next(e for e in second.exports() if e["id"] == export_id)
        assert row["status"] == "pending_delete"
        handle, _n, error = second.open_download(export_id)
        assert handle is None and error

        # Clearing the switch lets the retry succeed; the file is gone for good.
        second.db.set_settings({"delete_always_fail": "0"})
        second.retry_cleanup_now()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = next(e for e in second.exports() if e["id"] == export_id)
            if row["status"] == "deleted":
                break
            time.sleep(0.05)
        assert row["status"] == "deleted"
        assert not (second.export_dir / row["stored_name"]).exists()
    finally:
        second.shutdown()


def test_startup_sweeps_orphan_files_and_demotes_missing_files(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    first.start_workers()
    try:
        upload(first, CSV_SPRING)
        job = first.create_job("B1")
        wait_for(first, job)
        stored = first.exports()[0]["stored_name"]
    finally:
        first.shutdown()

    # Simulate an interrupted delete that left the file behind and an
    # interrupted publication that left a temp file.
    assert (data_dir / "exports" / stored).unlink() is None
    (data_dir / "exports" / "orphan.csv").write_text("leaked", encoding="utf-8")
    (data_dir / "tmp" / "half-written.csv").write_text("leaked", encoding="utf-8")

    second = Portal(data_dir)
    second.start_workers()
    try:
        assert not (data_dir / "exports" / "orphan.csv").exists()
        assert not list((data_dir / "tmp").glob("*.csv"))
        row = second.exports()[0]
        assert row["status"] == "pending_delete"
        handle, _n, error = second.open_download(row["id"])
        assert handle is None
    finally:
        second.shutdown()


def test_restart_fails_interrupted_jobs_and_keeps_queued(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    try:
        upload(first, CSV_SPRING)
        # Simulate a crash while the generator held a job mid-run.
        with first.db.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs(batch, status, requested_at) VALUES ('B1', 'running', ?)",
                ("2026-01-01T00:00:00+00:00",),
            )
            conn.execute(
                "INSERT INTO jobs(batch, status, requested_at) VALUES ('B2', 'queued', ?)",
                ("2026-01-01T00:00:00+00:00",),
            )
    finally:
        first.shutdown()

    second = Portal(data_dir)
    second.start_workers()
    try:
        jobs = {job["id"]: job for job in second.jobs()}
        crashed = next(j for j in jobs.values() if j["batch"] == "B1")
        queued = next(j for j in jobs.values() if j["batch"] == "B2")
        assert crashed["status"] == "failed"
        assert "重启" in crashed["message"]

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            queued = next(j for j in second.jobs() if j["batch"] == "B2")
            if queued["status"] in ("completed", "blocked", "failed"):
                break
            time.sleep(0.05)
        assert queued["status"] == "completed"
    finally:
        second.shutdown()


def test_download_blocks_text_withdrawn_before_export(portal):
    """Text ingested under another id must take an export down after revocation.

    This is the adversarial ordering: the duplicate copy is exported while its
    owner still has consent, then the real owner withdraws.
    """
    upload(
        portal,
        "batch,respondent_id,text\n"
        "B1,R-A,同一段访谈原文\n"
        "B1,R-B,另一段访谈原文\n"
        "B1,R-OTHER,同一段访谈原文\n",
    )
    job = portal.create_job("B1")
    wait_for(portal, job)
    export_id = portal.exports()[0]["id"]

    handle, _n, error = portal.open_download(export_id)
    assert error is None
    handle.close()

    portal.revoke("R-A")

    handle, _n, error = portal.open_download(export_id)
    assert handle is None and error
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = next(e for e in portal.exports() if e["id"] == export_id)
        if row["status"] == "deleted":
            break
        time.sleep(0.05)
    assert row["status"] == "deleted"


# ---------------------------------------------------------------------------
# redaction rules (遮蔽)
# ---------------------------------------------------------------------------
CSV_ORGS = """batch,respondent_id,text
B1,R1,我在康宁护理院做了三年护工
B1,R2,孩子考上了市第一中学
B1,R3,社区食堂的师傅手艺不错
B2,R4,康宁护理院对面的早餐铺
"""


def read_export(instance: Portal, export_id: int) -> str:
    handle, _name, error = instance.open_download(export_id)
    assert error is None and handle is not None
    content = handle.read().decode("utf-8")
    handle.close()
    return content


def test_redaction_masks_generated_csv_and_db_keeps_original(portal):
    upload(portal, CSV_ORGS)
    portal.add_redaction_rule("B1", "康宁护理院")
    job = portal.create_job("B1")
    wait_for(portal, job)

    content = read_export(portal, portal.exports()[0]["id"])
    assert "康宁护理院" not in content
    assert "[已遮蔽]" in content
    # 未登记的短语与其余已授权原话原样保留
    assert "市第一中学" in content
    assert "社区食堂的师傅手艺不错" in content
    # 库内保留原文，供撤回核查
    with portal.db.read() as conn:
        texts = [row["text"] for row in conn.execute("SELECT text FROM utterances")]
    assert any("康宁护理院" in text for text in texts)


def test_rule_locks_only_exports_containing_phrase(portal):
    upload(portal, "batch,respondent_id,text\nB1,R1,普通的访谈正文\n")
    wait_for(portal, portal.create_job("B1"))
    upload(portal, "batch,respondent_id,text\nB1,R2,提到了康宁护理院的访谈\n")
    wait_for(portal, portal.create_job("B1"))
    upload(portal, "batch,respondent_id,text\nB2,R9,提到康宁护理院的另一批次\n")
    wait_for(portal, portal.create_job("B2"))

    clean_b1, dirty_b1, other_batch = sorted(portal.exports(), key=lambda e: e["id"])
    result = portal.add_redaction_rule("B1", "康宁护理院")
    assert result["created"] is True
    assert result["locked_exports"] == 1

    # 含相关原文的旧导出立即禁下载并进入清理
    handle, _n, error = portal.open_download(dirty_b1["id"])
    assert handle is None and error
    row = next(e for e in portal.exports() if e["id"] == dirty_b1["id"])
    assert row["status"] in ("pending_delete", "deleted")
    # 不含该短语的同批次旧导出仍可下载
    assert "普通的访谈正文" in read_export(portal, clean_b1["id"])
    # 无关批次即使含该短语也照常可下载
    assert "康宁护理院" in read_export(portal, other_batch["id"])


def test_generation_after_rule_uses_new_rules(portal):
    upload(portal, CSV_ORGS)
    portal.add_redaction_rule("B1", "康宁护理院")
    portal.add_redaction_rule("B1", "市第一中学")
    wait_for(portal, portal.create_job("B1"))

    content = read_export(portal, portal.exports()[0]["id"])
    assert "康宁护理院" not in content
    assert "市第一中学" not in content
    assert content.count("[已遮蔽]") == 2
    assert "社区食堂" in content


def test_upload_after_rule_is_masked_on_next_generation(portal):
    upload(portal, "batch,respondent_id,text\nB1,R1,普通正文\n")
    portal.add_redaction_rule("B1", "康宁护理院")
    # 规则生效后上传的同批次语料，生成时同样按新规则遮蔽
    upload(portal, "batch,respondent_id,text\nB1,R2,后来去了康宁护理院工作\n")
    wait_for(portal, portal.create_job("B1"))

    content = read_export(portal, portal.exports()[0]["id"])
    assert "康宁护理院" not in content
    assert "[已遮蔽]" in content


def test_redaction_replaces_every_occurrence(portal):
    upload(portal, "batch,respondent_id,text\nB1,R1,康宁护理院和康宁护理院对面\n")
    portal.add_redaction_rule("B1", "康宁护理院")
    wait_for(portal, portal.create_job("B1"))

    content = read_export(portal, portal.exports()[0]["id"])
    assert "康宁护理院" not in content
    assert content.count("[已遮蔽]") == 2


def test_rule_save_with_unverifiable_export_locks_it_conservatively(portal):
    """内容核查失败的旧导出按命中处理：宁锁不错放，且清理失败也不可下载。"""
    upload(portal, CSV_ORGS)
    wait_for(portal, portal.create_job("B1"))
    export = portal.exports()[0]
    path = portal.export_dir / export["stored_name"]
    # 让规则保存时的文件核查失败：导出文件位置变成一个目录
    path.unlink()
    path.mkdir()

    result = portal.add_redaction_rule("B1", "康宁护理院")
    assert result["locked_exports"] == 1
    row = next(e for e in portal.exports() if e["id"] == export["id"])
    assert row["status"] == "pending_delete"

    # 规则已生效，旧内容绝不可下载
    handle, _n, error = portal.open_download(export["id"])
    assert handle is None and error
    # 清理失败（目标不是普通文件）时仍然不可下载
    portal.retry_cleanup_now()
    row = next(e for e in portal.exports() if e["id"] == export["id"])
    assert row["status"] == "pending_delete"
    handle, _n, error = portal.open_download(export["id"])
    assert handle is None and error


def test_rule_blocks_inflight_paused_job(client):
    """按旧规则制作的在途文件不得发布。"""
    test_client, instance = client
    body = "\n".join(f"B1,R{i},第{i}段访谈提到了康宁护理院的情况" for i in range(200))
    upload(instance, "batch,respondent_id,text\n" + body)
    instance.db.set_settings({"pause_generation": "1"})

    r = test_client.post("/generate", data={"batch": "B1"})
    assert r.status_code == 200
    deadline = time.monotonic() + 5
    paused = None
    while time.monotonic() < deadline:
        paused = next((j for j in instance.jobs() if j["status"] == "paused"), None)
        if paused:
            break
        time.sleep(0.05)
    assert paused is not None, "job never reached the pause gate"

    r = test_client.post("/redactions", data={"batch": "B1", "phrase": "康宁护理院"})
    assert r.status_code == 200
    assert "已登记遮蔽规则" in r.text

    instance.db.set_settings({"pause_generation": "0"})
    deadline = time.monotonic() + 5
    settled = None
    while time.monotonic() < deadline:
        settled = next(j for j in instance.jobs() if j["id"] == paused["id"])
        if settled["status"] == "blocked" and not list(instance.tmp_dir.glob("*")):
            break
        time.sleep(0.05)
    assert settled["status"] == "blocked"
    assert instance.exports() == []
    assert not list(instance.tmp_dir.glob("*"))


def test_queued_job_picks_up_new_rules(tmp_path):
    """排队中的任务尚未写出内容，启动时按新规则生成。"""
    instance = Portal(tmp_path / "data")
    try:
        upload(instance, CSV_ORGS)
        job_id = instance.create_job("B1")
        instance.add_redaction_rule("B1", "康宁护理院")
        instance.start_workers()
        wait_for(instance, job_id)

        content = read_export(instance, instance.exports()[0]["id"])
        assert "康宁护理院" not in content
        assert "[已遮蔽]" in content
    finally:
        instance.shutdown()


def test_duplicate_rule_is_idempotent(portal):
    upload(portal, CSV_ORGS)
    first = portal.add_redaction_rule("B1", "康宁护理院")
    assert first["created"] is True
    again = portal.add_redaction_rule("B1", "康宁护理院")
    assert again["created"] is False
    assert len(portal.redaction_rules()) == 1


def test_rule_validation(portal):
    from training_portal.service import ValidationError

    upload(portal, CSV_ORGS)
    with pytest.raises(ValidationError):
        portal.add_redaction_rule("B1", "   ")
    with pytest.raises(ValidationError):
        portal.add_redaction_rule("B1", "[已遮蔽]")
    with pytest.raises(ValidationError):
        portal.add_redaction_rule("B1", "换\n行")
    with pytest.raises(ValidationError):
        portal.add_redaction_rule("不存在的批次", "某短语")
    assert portal.redaction_rules() == []


def test_revocation_still_blocks_after_rules(portal):
    """撤回与遮蔽并存：同号、同文的阻断照常有效。"""
    upload(portal, CSV_ORGS)
    portal.add_redaction_rule("B1", "康宁护理院")
    wait_for(portal, portal.create_job("B1"))
    export_id = portal.exports()[0]["id"]

    result = portal.revoke("R3")
    assert result["affected_exports"] == 1
    handle, _n, error = portal.open_download(export_id)
    assert handle is None and "撤回" in error

    # 同号重传被拒
    result = upload(portal, "batch,respondent_id,text\nB1,R3,换新文本\n")
    assert result["skipped_revoked"] == 1
    # 同文换号重传也被拒（撤回时记录的是原文哈希）
    result = upload(portal, "batch,respondent_id,text\nB1,R9,社区食堂的师傅手艺不错\n")
    assert result["skipped_revoked"] == 1


def test_index_page_shows_redaction_section(client):
    test_client, instance = client
    upload(instance, CSV_ORGS)
    r = test_client.post("/redactions", data={"batch": "B1", "phrase": "康宁护理院"})
    assert r.status_code == 200
    assert "已登记遮蔽规则" in r.text

    r = test_client.get("/")
    assert r.status_code == 200
    assert "登记遮蔽短语" in r.text
    assert "康宁护理院" in r.text


def test_http_end_to_end_redaction_flow(client):
    test_client, instance = client
    r = test_client.post(
        "/upload",
        files={"file": ("orgs.csv", CSV_ORGS.encode("utf-8"), "text/csv")},
    )
    assert r.status_code == 200
    r = test_client.post("/generate", data={"batch": "B1"})
    assert r.status_code == 200
    job = wait_for(instance, instance.jobs()[0]["id"])
    assert job["status"] == "completed"
    old_export_id = instance.exports()[0]["id"]

    r = test_client.get(f"/download/{old_export_id}")
    assert r.status_code == 200
    assert "康宁护理院" in r.text

    r = test_client.post("/redactions", data={"batch": "B1", "phrase": "康宁护理院"})
    assert r.status_code == 200

    # 旧导出立即禁下载
    r = test_client.get(f"/download/{old_export_id}")
    assert r.status_code in (404, 410)

    # 重新生成：新导出按新规则遮蔽
    r = test_client.post("/generate", data={"batch": "B1"})
    assert r.status_code == 200
    job = wait_for(instance, instance.jobs()[0]["id"])
    assert job["status"] == "completed"
    new_export = instance.exports()[0]
    assert new_export["id"] != old_export_id
    r = test_client.get(f"/download/{new_export['id']}")
    assert r.status_code == 200
    assert "康宁护理院" not in r.text
    assert "[已遮蔽]" in r.text

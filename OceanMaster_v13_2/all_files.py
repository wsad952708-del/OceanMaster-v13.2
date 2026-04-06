from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

"""DES-004 (SPEC-002) TDD 測試：all_files.py 路徑安全與環境變數配置。

漁業科學意義：
    驗證 OceanMaster 專案檔案掃描工具符合 FDC-001/FDC-002 規範，
    確保 RFMO（WCPFC/IOTC）合規審計的 processing lineage 完整性。
"""

from pathlib import Path

import pytest

from all_files import (
    PathTraversalError,
    get_config,
    scan_project,
    validate_cpue,
    validate_nc_quality_flag,
    validate_path,
)


# ──────────────────────────────────────────────────────────────────
# T4-1: OCEANMASTER_ROOT 環境變數正確解析
# ──────────────────────────────────────────────────────────────────
class TestT4_1_EnvRootValid:
    def test_root_resolves_correctly(self, tmp_path, monkeypatch):
        """設定 OCEANMASTER_ROOT=/valid/path → root_path == Path('/valid/path').resolve()"""
        valid_dir = tmp_path / "oceanmaster_project"
        valid_dir.mkdir()
        monkeypatch.setenv("OCEANMASTER_ROOT", str(valid_dir))
        monkeypatch.delenv("OCEANMASTER_OUTPUT_DIR", raising=False)

        config = get_config()
        assert config["root_path"] == valid_dir.resolve()
        assert isinstance(config["root_path"], Path)

    def test_output_dir_from_env(self, tmp_path, monkeypatch):
        """設定 OCEANMASTER_OUTPUT_DIR → 正確解析。"""
        out_dir = tmp_path / "custom_output"
        out_dir.mkdir()
        monkeypatch.setenv("OCEANMASTER_OUTPUT_DIR", str(out_dir))
        monkeypatch.delenv("OCEANMASTER_ROOT", raising=False)

        config = get_config()
        assert config["output_dir"] == out_dir.resolve()


# ──────────────────────────────────────────────────────────────────
# T4-2: 未設定環境變數 → 使用 cwd 預設值且不 crash
# ──────────────────────────────────────────────────────────────────
class TestT4_2_NoEnvDefaults:
    def test_defaults_no_crash(self, monkeypatch):
        """無環境變數 → 使用 cwd 為基底的預設值。"""
        monkeypatch.delenv("OCEANMASTER_ROOT", raising=False)
        monkeypatch.delenv("OCEANMASTER_OUTPUT_DIR", raising=False)

        config = get_config()
        expected_root = (Path.cwd() / "OceanMaster_v13_2").resolve()
        expected_output = (Path.cwd() / "output").resolve()
        assert config["root_path"] == expected_root
        assert config["output_dir"] == expected_output

    def test_config_returns_path_objects(self, monkeypatch):
        """回傳值均為 Path 物件。"""
        monkeypatch.delenv("OCEANMASTER_ROOT", raising=False)
        monkeypatch.delenv("OCEANMASTER_OUTPUT_DIR", raising=False)

        config = get_config()
        assert isinstance(config["root_path"], Path)
        assert isinstance(config["output_dir"], Path)


# ──────────────────────────────────────────────────────────────────
# T4-3: 路徑遍歷攻擊 → PathTraversalError
# ──────────────────────────────────────────────────────────────────
class TestT4_3_PathTraversal:
    def test_root_with_dotdot_raises(self, monkeypatch):
        """OCEANMASTER_ROOT 含 '../../../etc' → PathTraversalError。"""
        monkeypatch.setenv("OCEANMASTER_ROOT", "../../../etc")
        monkeypatch.delenv("OCEANMASTER_OUTPUT_DIR", raising=False)

        with pytest.raises(PathTraversalError, match=r"\.\."):
            get_config()

    def test_output_with_dotdot_raises(self, monkeypatch):
        """OCEANMASTER_OUTPUT_DIR 含 '..' → PathTraversalError。"""
        monkeypatch.delenv("OCEANMASTER_ROOT", raising=False)
        monkeypatch.setenv("OCEANMASTER_OUTPUT_DIR", "/tmp/../../../etc/shadow")

        with pytest.raises(PathTraversalError, match=r"\.\."):
            get_config()

    def test_validate_path_dotdot(self):
        """validate_path 偵測 '..' 並拋出 PathTraversalError。"""
        with pytest.raises(PathTraversalError):
            validate_path("../../../etc/passwd")

    def test_resolved_path_no_dotdot(self, tmp_path):
        """resolve() 後路徑不含 '..' 片段。"""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        resolved = validate_path(safe_dir, must_exist=True)
        assert ".." not in resolved.parts


# ──────────────────────────────────────────────────────────────────
# T4-4: 原始碼不含 'c:\Users\user'
# ──────────────────────────────────────────────────────────────────
class TestT4_4_NoWindowsUserPath:
    @pytest.fixture
    def source_code(self):
        src = Path(__file__).resolve().parent.parent / "all_files.py"
        if not src.exists():
            src = Path(__file__).resolve().parent / "all_files.py"
        return src.read_text(encoding="utf-8")

    def test_no_backslash_user_path(self, source_code):
        assert "c:\\users\\user" not in source_code.lower()

    def test_no_forward_slash_user_path(self, source_code):
        assert "c:/users/user" not in source_code.lower()

    def test_no_raw_string_user_path(self, source_code):
        assert r"c:\\users\\" not in source_code.lower(), "Raw-string Windows user path detected"


# ──────────────────────────────────────────────────────────────────
# T4-5: scan_project 回傳檔案清單
# ──────────────────────────────────────────────────────────────────
class TestT4_5_ScanProject:
    def test_scan_returns_list(self, tmp_path, monkeypatch):
        """scan_project 回傳非空檔案清單。"""
        (tmp_path / "data.csv").write_text("col\n1")
        monkeypatch.setenv("OCEANMASTER_ROOT", str(tmp_path))
        result = scan_project(tmp_path)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_scan_excludes_hidden(self, tmp_path, monkeypatch):
        """scan_project 排除隱藏檔。"""
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.py").write_text("")
        monkeypatch.setenv("OCEANMASTER_ROOT", str(tmp_path))
        result = scan_project(tmp_path)
        names = [p.name for p in result]
        assert ".hidden" not in names


class TestT4_6_ValidateCpue:
    def test_cpue_negative_raises(self):
        """CPUE 為負值 → ValueError（FDC-002 科學合理性）。"""
        with pytest.raises(ValueError, match="CPUE"):
            validate_cpue(-1.0)

    def test_cpue_valid(self):
        assert validate_cpue(12.5) == 12.5


class TestT4_7_NcQualityFlag:
    def test_invalid_flag_raises(self):
        with pytest.raises(ValueError):
            validate_nc_quality_flag(99)

    def test_valid_flag(self):
        assert validate_nc_quality_flag(1) == 1
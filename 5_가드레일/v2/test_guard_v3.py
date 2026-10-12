# -*- coding: utf-8 -*-
"""가드레일 v3 추가 단계(G1 범위 판정, G2 적용 대상 판정, G3 완화) 단위 테스트. 가짜 LLM 사용.
사용: python -m unittest test_guard_v3 -v
"""
import json, unittest
from guard import (Guard, ScopeClassifier, subject_out_of_scope, EVIDENCE_SYSTEM, EVIDENCE_SYSTEM_RELAXED,
                   ANSWER, REFUSE_SCOPE, REFUSE_SUBJECT, REFUSE_NO_EVIDENCE, SEE_RAW_NOTE)
from test_guard import ARTS, ORG, DELEG, ev

class FakeLLM:
    """추출 호출과 범위 분류 호출을 schema 로 구분해 각각 정해진 답을 준다."""
    def __init__(self, extract, scope=None): self.extract, self.scope, self.systems = extract, scope, []
    def __call__(self, system, user, schema=None):
        self.systems.append(system)
        out = self.scope if (schema and "confidence" in schema.get("properties", {})) else self.extract
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)

Q_IN = "육아휴직은 얼마나 할 수 있어요?"
GOOD = ev([(1, "② 육아휴직의 기간은 자녀 1명에 대하여 3년 이내로 한다.")])

class TestG1Scope(unittest.TestCase):
    def test_other_rule_refuses_before_extraction(self):
        llm = FakeLLM(GOOD, {"scope": "다른규정", "other_rule": "보수규정", "confidence": 0.9})
        r = Guard(llm, scope_classifier=ScopeClassifier(llm)).respond("성과급 언제 나와요?", ORG, ARTS, DELEG)
        self.assertEqual(r.decision, REFUSE_SCOPE); self.assertIn("보수규정 소관", r.text)
        self.assertIn("가장 가까운 인사규정 조항", r.text); self.assertIn(SEE_RAW_NOTE, r.text)
        self.assertEqual(len(llm.systems), 1)  # 추출은 호출되지 않음

    def test_unrelated_refuses(self):
        llm = FakeLLM(GOOD, {"scope": "무관", "other_rule": "", "confidence": 0.8})
        r = Guard(llm, scope_classifier=ScopeClassifier(llm)).respond("탕비실 원두 누가 채워요?", ORG, ARTS, DELEG)
        self.assertEqual(r.decision, REFUSE_SCOPE); self.assertIn("다루지 않는 사항", r.text)

    def test_low_confidence_out_of_scope_still_refuses(self):
        # 애매해도 통과시키지 않는다 (v3 정책)
        llm = FakeLLM(GOOD, {"scope": "다른규정", "other_rule": "복무규정", "confidence": 0.3})
        r = Guard(llm, scope_classifier=ScopeClassifier(llm)).respond("연차 몇 개 남았어요?", ORG, ARTS, DELEG)
        self.assertEqual(r.decision, REFUSE_SCOPE)

    def test_in_scope_proceeds_to_extraction(self):
        llm = FakeLLM(GOOD, {"scope": "인사규정", "other_rule": "", "confidence": 0.95})
        r = Guard(llm, scope_classifier=ScopeClassifier(llm)).respond(Q_IN, ORG, ARTS, DELEG)
        self.assertEqual(r.decision, ANSWER); self.assertEqual(r.trace["scope"]["scope"], "인사규정")

    def test_malformed_scope_output_passes_through(self):
        llm = FakeLLM(GOOD, "JSON 아님")
        r = Guard(llm, scope_classifier=ScopeClassifier(llm)).respond(Q_IN, ORG, ARTS, DELEG)
        self.assertEqual(r.decision, ANSWER); self.assertIn("parse_error", r.trace["scope"])

    def test_precomputed_scope_result_is_used_without_call(self):
        llm = FakeLLM(GOOD)
        r = Guard(llm).respond("성과급?", ORG, ARTS, DELEG, scope_result={"scope": "다른규정", "other_rule": "보수규정", "confidence": 1.0})
        self.assertEqual(r.decision, REFUSE_SCOPE); self.assertEqual(llm.systems, [])

    def test_prompt_with_and_without_lists(self):
        from guard import build_scope_prompt
        a = build_scope_prompt("q", ORG, ["제1장 총칙"], ["보수규정"], True); b = build_scope_prompt("q", ORG, ["제1장 총칙"], ["보수규정"], False)
        self.assertIn("제1장 총칙", a); self.assertNotIn("제1장 총칙", b); self.assertIn("[질문]", b)

class TestG2Subject(unittest.TestCase):
    def test_executive_question_refused(self):
        llm = FakeLLM(GOOD)
        r = Guard(llm, subject_check=True).respond("임원도 건강 때문에 휴직되나요?", ORG, ARTS, DELEG, scope_article_text="제2조(적용범위) 직원의 인사에 관하여 이 규정에 따른다.")
        self.assertEqual(r.decision, REFUSE_SUBJECT); self.assertIn("임원", r.text); self.assertEqual(llm.systems, [])

    def test_executive_included_by_scope_article_passes(self):
        llm = FakeLLM(GOOD)
        r = Guard(llm, subject_check=True).respond("임원도 휴직되나요?", ORG, ARTS, DELEG, scope_article_text="제2조(적용범위) 이 규정은 임원을 포함한 전 직원에게 적용한다.")
        self.assertEqual(r.decision, ANSWER)

    def test_negated_scope_article_still_refuses(self):
        """'적용하지 않는다'·'포함하지 아니한다'·'제외한다'는 포함이 아니다 (2026-10-12 검토에서 발견)."""
        for art in ("제2조(적용범위) 이 규정은 임원에게는 적용하지 않는다.",
                    "제2조(적용범위) 직원이란 정규직을 말하며 계약직은 포함하지 아니한다.",
                    "제2조(적용범위) 임원은 이 규정의 적용 대상에서 제외한다."):
            q = "임원도 휴직되나요?" if "임원" in art else "계약직도 승진 돼요?"
            self.assertIsNotNone(subject_out_of_scope(q, art), art)

    def test_plain_employee_question_unaffected(self):
        self.assertIsNone(subject_out_of_scope("직원 육아휴직 기간은?", ""))
        self.assertEqual(subject_out_of_scope("계약직도 승진 돼요?", ""), "계약직")

    def test_no_scope_article_still_refuses_by_rule(self):
        llm = FakeLLM(GOOD)
        r = Guard(llm, subject_check=True).respond("퇴직자도 징계 받나요?", ORG, ARTS, DELEG, scope_article_text="")
        self.assertEqual(r.decision, REFUSE_SUBJECT)

class TestG3Relaxed(unittest.TestCase):
    def test_relaxed_prompt_is_used_only_when_enabled(self):
        llm = FakeLLM(GOOD)
        Guard(llm).respond(Q_IN, ORG, ARTS, DELEG); Guard(llm, relaxed=True).respond(Q_IN, ORG, ARTS, DELEG)
        self.assertEqual(llm.systems[0], EVIDENCE_SYSTEM); self.assertEqual(llm.systems[1], EVIDENCE_SYSTEM_RELAXED)

    def test_relaxed_keeps_verbatim_rule(self):
        self.assertIn("한 글자도 바꾸지 않고", EVIDENCE_SYSTEM_RELAXED)
        llm = FakeLLM(ev([(1, "육아휴직은 자녀 한 명당 최대 3년까지 가능합니다.")]))
        r = Guard(llm, relaxed=True).respond(Q_IN, ORG, ARTS, DELEG)
        self.assertEqual(r.decision, REFUSE_NO_EVIDENCE)

if __name__ == "__main__":
    unittest.main(verbosity=2)

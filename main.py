"""
Agentic RAG System - Main orchestration using LangGraph
"""

import json
import re
from datetime import date
from typing import Literal, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState

from tools.rag_tool import create_rag_tool
from tools.weather_tool import create_weather_tool
from tools.calendar_tool import create_calendar_tool
from tools.holiday_tool import create_holiday_tool

print("Initializing Agentic RAG System...")
print("=" * 60)

print("Loading tools...")
rag_tool_instance = create_rag_tool()
weather_tool_instance = create_weather_tool()
calendar_tool_instance = create_calendar_tool()
holiday_tool_instance = create_holiday_tool()
print("✓ All tools loaded successfully")


@tool
def search_course_materials(query: str) -> str:
    """Search Information Retrieval course materials (RAG)."""
    return rag_tool_instance.search(query)


@tool
def get_weather(location: str, on_date: str = None) -> str:
    """Weather (external API). If on_date provided (YYYY-MM-DD), returns forecast if available."""
    return weather_tool_instance.get_weather(location, on_date=on_date)


@tool
def check_calendar(query: str = "upcoming") -> str:
    """
    Calendar (internal tool). If query contains 'next exam', returns JSON for chaining.
    """
    q = (query or "").lower().strip()
    if "next exam" in q:
        return calendar_tool_instance.get_next_exam_json()

    if "deadline" in q:
        return calendar_tool_instance.get_next_deadline()
    elif "exam" in q or "test" in q:
        return calendar_tool_instance.check_specific_event("exam")
    else:
        return calendar_tool_instance.get_upcoming_events()


@tool
def get_next_holiday(country_code: str = "IL") -> str:
    """External API: next upcoming holiday after today's date."""
    return holiday_tool_instance.get_next_holiday(country_code=country_code, today_iso=date.today().isoformat())


@tool
def get_public_holidays(year: int, country_code: str = "IL") -> str:
    """External API: list holidays in a given year."""
    return holiday_tool_instance.get_holidays(year=year, country_code=country_code)


@tool
def is_public_holiday(date_str: str, country_code: str = "IL") -> str:
    """External API: check if a specific date is a holiday."""
    return holiday_tool_instance.is_holiday(date_str=date_str, country_code=country_code)


tools = [
    search_course_materials,
    get_weather,
    check_calendar,
    get_next_holiday,
    get_public_holidays,
    is_public_holiday,
]
tool_dict = {t.name: t for t in tools}

print("Initializing Llama 3.1 model...")
llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://localhost:11434")
llm_with_tools = llm.bind_tools(tools)
print("✓ Model initialized with tool capabilities")


class AgentState(MessagesState):
    pass


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _user_requested_weather_on_exam_day(user_text: str) -> bool:
    t = user_text.lower()
    return ("exam" in t) and ("weather" in t) and (("that day" in t) or ("specific day" in t) or ("exam day" in t))


def _extract_date_from_calendar_json(tool_text: str) -> Optional[str]:
    try:
        obj = json.loads(tool_text)
        if obj.get("found") and isinstance(obj.get("date"), str):
            if re.match(r"^\d{4}-\d{2}-\d{2}$", obj["date"]):
                return obj["date"]
    except Exception:
        pass
    return None


def _is_social_or_useless(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    social = {
        "you're welcome!", "you’re welcome!", "welcome!", "no problem", "np",
        "ok", "okay", "sure", "thanks", "thank you", "great", "cool"
    }
    if t in social:
        return True
    if len(t) <= 4 and t in {"ok", "k", "sure", "yes", "no"}:
        return True
    return False


def _unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _format_exam_json(calendar_out: str) -> Optional[str]:
    """
    If calendar_out looks like JSON for next exam, format it nicely.
    Returns formatted string or None if not JSON / not matching expected schema.
    """
    try:
        obj = json.loads(calendar_out)
        if isinstance(obj, dict) and obj.get("found") is True and obj.get("type") == "exam":
            title = obj.get("title", "Exam")
            d = obj.get("date", "")
            tm = obj.get("time", "")
            return f"Next exam: {title}\nDate: {d} {tm}".strip()
    except Exception:
        return None
    return None


def _extract_tool_outputs(messages) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns (weather_out, calendar_out, holiday_out, rag_out)
    """
    weather_out = None
    calendar_out = None
    holiday_out = None
    rag_out = None

    for m in messages:
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", "") or ""
            content = (m.content or "").strip()
            if name == "get_weather":
                weather_out = content
            elif name == "check_calendar":
                calendar_out = content
            elif name == "search_course_materials":
                rag_out = content
            elif name in {"get_next_holiday", "get_public_holidays", "is_public_holiday"}:
                holiday_out = content

    return weather_out, calendar_out, holiday_out, rag_out


def _llm_course_explanation(user_input: str, rag_evidence: str) -> str:
    """
    Generate a clean course explanation using LLM, grounded on retrieved evidence.
    This is used especially in multi-tool queries so we don't dump raw chunks as the explanation.
    """
    sys = SystemMessage(content=(
        "You are an expert Information Retrieval tutor.\n"
        "Task: Answer ONLY the course/theory part of the user's request.\n"
        "Use the provided evidence as grounding. If evidence is weak, answer with best effort but stay general.\n"
        "Output format:\n"
        "1) Main explanation (2–4 sentences)\n"
        "2) Example / formula if relevant (1–3 lines)\n"
        "3) Practical connection (1–2 sentences)\n"
        "Do NOT mention tools. Do NOT paste the evidence."
    ))
    human = HumanMessage(content=f"USER QUESTION:\n{user_input}\n\nEVIDENCE:\n{rag_evidence}")
    try:
        resp = llm.invoke([sys, human])
        text = (getattr(resp, "content", "") or "").strip()
        return text if text else "I couldn't generate a course explanation from the evidence."
    except Exception:
        return "I couldn't generate a course explanation from the evidence."


def build_final_answer_multi(
    user_input: str,
    weather_out: Optional[str],
    calendar_out: Optional[str],
    holiday_out: Optional[str],
    rag_out: Optional[str],
) -> str:
    """
    Multi-tool answer:
    - Non-RAG sections come directly from tool outputs (deterministic)
    - Course explanation is an LLM-generated summary grounded in RAG evidence
    - Evidence section includes the raw RAG chunks
    """
    parts = []

    if weather_out:
        parts.append("**Weather**\n" + weather_out)

    if calendar_out:
        pretty = _format_exam_json(calendar_out)
        parts.append("**Schedule / Exams**\n" + (pretty if pretty else calendar_out))

    if holiday_out:
        parts.append("**Holidays**\n" + holiday_out)

    if rag_out:
        course_expl = _llm_course_explanation(user_input=user_input, rag_evidence=rag_out)
        parts.append("**Course explanation**\n" + course_expl)
        parts.append("**Evidence (retrieved chunks)**\n" + rag_out)

    return "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()


def agent_node(state: AgentState):
    messages = state["messages"]
    today = date.today().isoformat()

    system_prompt = SystemMessage(content=f"""
You are a helpful assistant for a student. You can use tools.

CONTEXT:
- Today's date is {today}.
- Default location is Haifa, Israel.

CRITICAL RULES:
- If the user asks multiple things, answer ALL of them in one response.
- After tool call(s), restate results clearly in your answer.
- Do NOT respond with generic filler when the user asked a question.

LOCATION:
- If user asks weather without specifying a city, assume Haifa, Israel.

HOLIDAYS:
- "next holiday" -> get_next_holiday
- list holidays -> get_public_holidays(year)
- check specific date -> is_public_holiday(date)

RAG:
- Use search_course_materials for IR concepts.
- Summarize: 2–4 sentences + example/formula if relevant + practical connection.

HONESTY:
- Never invent dates, weather, holidays, or events.
- If forecast not available for that date, say so.
""".strip())

    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    user_text = _last_user_text(messages)

    outputs = []

    exam_date: Optional[str] = None
    saw_weather_call_missing_date = False
    weather_location_used: Optional[str] = None

    tool_calls = getattr(last_message, "tool_calls", None) or []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"   Arguments: {tool_args}")

        # Guard: reject placeholder locations for weather
        if tool_name == "get_weather":
            loc = str(tool_args.get("location", "")).strip()
            on_date = tool_args.get("on_date", None)

            bad_locs = {"current location", "my location", "here", "now", "today", "local", "location"}
            if loc.lower() in bad_locs or len(loc) < 2:
                tool_result = "Error: No valid city provided. Please provide a real city (e.g., 'Haifa, Israel')."
            else:
                weather_location_used = loc
                if not on_date:
                    saw_weather_call_missing_date = True
                tool_result = tool_dict[tool_name].invoke(tool_args)
        else:
            tool_result = tool_dict[tool_name].invoke(tool_args)

        preview = str(tool_result)[:160].replace("\n", " ")
        print(f"   Result preview: {preview}...")

        if tool_name == "check_calendar":
            maybe_date = _extract_date_from_calendar_json(str(tool_result))
            if maybe_date:
                exam_date = maybe_date

        outputs.append(
            ToolMessage(
                content=str(tool_result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )
        )

    # Auto-chaining: weather on exam day if model forgot on_date
    if _user_requested_weather_on_exam_day(user_text) and exam_date and weather_location_used and saw_weather_call_missing_date:
        print(f"\n🔁 Auto-chaining: Fetching forecast for exam date {exam_date}...")
        chained = get_weather.invoke({"location": weather_location_used, "on_date": exam_date})
        outputs.append(
            ToolMessage(
                content=str(chained),
                name="get_weather",
                tool_call_id="auto_chained_exam_weather"
            )
        )

    return {"messages": outputs}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    return "tools" if tool_calls else "end"


print("Building LangGraph workflow...")
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
app = workflow.compile()

print("✓ LangGraph workflow compiled successfully")
print("=" * 60)
print("\nSystem ready! 🚀\n")


def chat(user_input: str):
    print(f"\n{'=' * 60}")
    print(f"USER: {user_input}")
    print(f"{'=' * 60}")

    state = {"messages": [HumanMessage(content=user_input)]}
    result = app.invoke(state)

    messages = result["messages"]
    final_message = messages[-1]
    llm_response = (getattr(final_message, "content", "") or "").strip()

    # Detect tool usage and which tools were used
    tool_names = []
    for m in messages:
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", "") or ""
            if name:
                tool_names.append(name)
    unique_tools = _unique_preserve_order(tool_names)
    used_tools = len(unique_tools) > 0

    weather_out, calendar_out, holiday_out, rag_out = _extract_tool_outputs(messages)

    # HYBRID POLICY:
    # - If only RAG used: show LLM answer (nice), optionally add evidence if you want.
    # - If multiple tools used and RAG used: show deterministic non-RAG + LLM course explanation + evidence.
    # - If non-RAG only: show deterministic tool output.
    if not used_tools:
        response = llm_response

    elif len(unique_tools) == 1 and unique_tools[0] == "search_course_materials":
        # Single RAG question -> prefer LLM summary
        response = llm_response if not _is_social_or_useless(llm_response) else (rag_out or llm_response)

    elif rag_out:
        # Multi-tool with RAG -> deterministic sections + LLM course explanation + evidence
        response = build_final_answer_multi(
            user_input=user_input,
            weather_out=weather_out,
            calendar_out=calendar_out,
            holiday_out=holiday_out,
            rag_out=rag_out,
        )

    else:
        # Multi-tool without RAG -> deterministic aggregation from tools
        parts = []
        if weather_out:
            parts.append("**Weather**\n" + weather_out)
        if calendar_out:
            pretty = _format_exam_json(calendar_out)
            parts.append("**Schedule / Exams**\n" + (pretty if pretty else calendar_out))
        if holiday_out:
            parts.append("**Holidays**\n" + holiday_out)
        response = "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()
        if not response:
            response = llm_response

    # Final fallback: if model was useless and deterministic is empty, show last tool output
    if _is_social_or_useless(response):
        for m in reversed(messages):
            if isinstance(m, ToolMessage) and (m.content or "").strip():
                response = m.content.strip()
                break

    print(f"\n{'=' * 60}")
    print(f"ASSISTANT: {response}")
    print(f"{'=' * 60}\n")

    return response


if __name__ == "__main__":
    print("Agentic RAG System - Interactive Mode")
    print("Ask me about:")
    print("  • Course materials (e.g., 'What is PageRank?')")
    print("  • Weather (e.g., 'What's the weather in Haifa today?')")
    print("  • Your schedule (e.g., 'When is my next exam?')")
    print("  • Holidays (e.g., 'When is the next holiday in Israel?')")
    print("\nType 'quit' to exit\n")

    while True:
        try:
            user_input = input("YOU: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nGoodbye! 👋\n")
                break
            chat(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            import traceback
            traceback.print_exc()
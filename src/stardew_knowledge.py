"""
Curated Stardew Valley knowledge base with retrieval.
Provides factual context to the LLM to reduce hallucinations.
"""

# Words that carry no topic signal for fact matching. Without this list the
# generic word "fish" matched every fact that mentions any fish, and question
# words (where/when/what) matched nearly everything.
_FACT_STOPWORDS = {
    "fish", "fishes", "fishing", "catch", "caught", "get", "find", "found",
    "where", "when", "what", "how", "who", "the", "and", "can", "you",
    "farm", "farming", "game", "stardew", "valley", "use", "using", "make",
    "made", "grow", "grown", "does", "are", "for", "with", "from", "into",
}

STARDEW_FACTS = {
    "coal": [
        "Coal is found in the Mines starting at floor 41 and deeper.",
        "Coal can be obtained by breaking barrels and crates in the Mines.",
        "Each coal ore found in the Mines requires no refinement - it is already coal.",
        "Coal is used for smelting (combining with other items at the Forge).",
        "Coal is NOT found in the Quarry or other mines.",
        "You cannot 'farm' coal by planting seeds - it only comes from mining.",
    ],
    "spring": [
        "Spring crops: Parsnip, Cauliflower, Green Bean, Strawberry.",
        "Parsnips grow in 4-5 days.",
        "Cauliflower grows in 12-14 days.",
        "Green Beans grow in 10-14 days and regrow every 3 days.",
        "Strawberries grow in 8-9 days.",
        "Spring lasts from day 1 to day 28.",
    ],
    "summer": [
        "Summer crops: Tomato, Melon, Pepper, Corn, Sunflower, Hops, Blueberry.",
        "Melons grow in 12-14 days.",
        "Corn grows in 14 days and regrows every 3 days.",
        "Blueberry grows in 13 days and regrows every 3 days.",
        "Summer lasts from day 1 to day 28.",
    ],
    "fall": [
        "Fall crops: Corn, Yam, Pumpkin, Cranberry, Eggplant, Sunflower.",
        "Pumpkin grows in 13-14 days.",
        "Cranberry grows in 7-8 days and regrows every 5 days.",
        "Yam grows in 10-11 days.",
        "Fall lasts from day 1 to day 28.",
    ],
    "winter": [
        "Winter crops: Winter Seeds, Winter Roots, Snow Yam, Crocus, Daffodil, Sagebrush.",
        "Winter Seeds grow in 7 days and regrow every 7 days.",
        "Winter lasts from day 1 to day 28.",
        "Most traditional crops cannot grow in Winter on the farm.",
    ],
    "fertilizer": [
        "Basic Fertilizer increases harvest yield by 25%.",
        "Quality Fertilizer increases yield by 25% AND increases quality of produce.",
        "Speed-Gro fertilizer reduces growth time by 1 day.",
        "Deluxe Speed-Gro reduces growth time by 2 days.",
        "Fertilizer is placed in soil BEFORE planting seeds.",
        "Fertilizer is made at the silo or bought from Pierre's General Store.",
    ],
    "irrigation": [
        "Sprinklers automatically water crops.",
        "Basic Sprinkler waters crops in a '+' shape (4 tiles).",
        "Quality Sprinkler waters crops in a 3x3 square (8 tiles).",
        "Iridium Sprinkler waters crops in a 5x5 square (24 tiles).",
        "Without sprinklers, you must manually water with a watering can each day.",
    ],
    "bundles": [
        "Bundles are in the Community Center.",
        "Completing bundles rewards items like parrot perches, golden clock, and bridge repairs.",
        "There are 6 bundle rooms: Pantry, Crafts, Pantry, Fish Tank, Bulletin Board, Vault.",
        "Completing all bundles restores the Community Center.",
    ],
    "marriage": [
        "You can marry any bachelor (Alex, Elliott, Harvey, Sam, Sebastian, Shane) or bachelorette (Abigail, Emily, Haley, Leah, Maru, Penny).",
        "Marriage requires 10 hearts with the character.",
        "Give gifts to increase relationship. Most characters prefer 1-2 gifts per week.",
        "After marriage, they move into your farmhouse.",
        "You can have children after marriage.",
    ],
    "mining": [
        "The Mines have 120 floors. Floors 1-40 are accessible initially.",
        "Use a pickaxe to break rocks and collect ores (copper, iron, gold, iridium).",
        "Copper ore is on floors 1-40.",
        "Iron ore is on floors 41-80.",
        "Gold ore is on floors 41-120.",
        "Iridium ore is on floors 81-120.",
        "Take stairs down, avoid or defeat monsters.",
    ],
    # One fact per species so per-fact matching never mixes them up. Values
    # cross-checked against the wiki infoboxes (Location/Season/Time/Weather).
    "fishing": [
        "Bullhead is found in the Mountain Lake, in any season, any weather, and any time of day.",
        "Flounder is found in the Ocean and on Ginger Island, during Spring and Summer, from 6am to 8pm, in any weather.",
        "Pike can be caught in the river in Pelican Town or Cindersap Forest and in the pond in Cindersap Forest.",
        "Pike can also be caught on Riverland Farm and in the large pond on Forest Farm.",
        "Pike can be caught in Summer and Winter.",
        "Pike can be caught at any time of day and in any weather.",
    ],
    "quality": [
        "Crop quality levels: Regular, Silver, and Gold.",
        "Higher quality produces sell for more money.",
        "Quality Fertilizer increases the chance of higher quality crops.",
        "Tilled soil at higher farming levels produces higher quality crops naturally.",
    ],
}

def retrieve_stardew_facts(query: str, max_results: int = 3) -> str:
    """
    Retrieve relevant Stardew facts based on query keywords.
    Returns formatted context string to inject into LLM prompt.

    Matching is per-fact: a fact is included only when it shares a content
    token with the query. Topic-level matching (the old behaviour) pulled in
    every sibling fact, so a Pike-only "fishing" topic hijacked all fishing
    questions. Stopwords and generic words (fish, catch, farm, where, when)
    are ignored so facts are chosen by their distinctive content only.
    """
    query_lower = query.lower()
    query_tokens = {
        token
        for token in ''.join(ch if ch.isalnum() else ' ' for ch in query_lower).split()
        if len(token) > 2 and token not in _FACT_STOPWORDS
    }
    if not query_tokens:
        return ""

    scored_facts: list[tuple[int, str, str]] = []
    for topic, facts in STARDEW_FACTS.items():
        for fact in facts:
            fact_tokens = {
                token
                for token in ''.join(ch if ch.isalnum() else ' ' for ch in fact.lower()).split()
                if len(token) > 2 and token not in _FACT_STOPWORDS
            }
            overlap = len(query_tokens & fact_tokens)
            if overlap > 0:
                scored_facts.append((overlap, topic, fact))

    if not scored_facts:
        return ""

    # Return top facts as context, preferring the strongest overlap.
    scored_facts.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    context_lines = [f"**{topic}**: {fact}" for _, topic, fact in scored_facts[:max_results]]
    return "\n".join(context_lines)


def get_system_prompt() -> str:
    """
    Generate a system prompt that makes the model more reliable for Stardew Valley.
    """
    return """You are a knowledgeable Stardew Valley farming assistant.

Your role:
- Answer ONLY questions about Stardew Valley game mechanics, crops, NPCs, and farming.
- Provide accurate, concise responses based on the game's actual mechanics.
- If you don't know the answer, say "I'm not sure about that - you might want to check the Stardew Valley wiki."
- Do NOT make up crop growth times, mining depths, NPC preferences, or game mechanics.
- Do NOT confuse real-world farming with Stardew Valley gameplay.

Stardew Valley facts:
- Crops grow over a specific number of GAME DAYS (not real-world days).
- Fertilizer placement timing matters: apply BEFORE planting.
- NPCs have specific preferences - check their loved/liked/disliked gifts.
- Mining requires proper tools and floor progression.
- Fish live in specific locations (river, ocean, lake, mountain, swamp)
  and are only catchable in certain seasons, times, and weather.
- Quality crops sell for significantly more money.

When answering:
1. Answer in 1-2 short sentences.
2. State the core fact first.
3. Do not quote the wiki text or repeat it verbatim.
4. Be specific about game mechanics (not general agriculture advice).
"""

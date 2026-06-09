#!/usr/bin/env python3
"""
UGC Brand Deal AI Agent
Searches for brands, extracts emails, sends personalized outreach, logs results.
"""

import argparse
import csv
import json
import logging
import re
import smtplib
import sys
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

RESULTS_HEADERS = [
    "business_name", "website", "email",
    "status", "sent_at", "subject", "message_preview",
]

CONTACT_PATHS = ["/contact", "/contact-us", "/about", "/about-us", "/reach-us", "/info"]
THROWAWAY_DOMAINS = {
    "example.com", "test.com", "email.com", "domain.com",
    "yoursite.com", "yourdomain.com", "sentry.io", "wixpress.com",
}
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


class Config:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.your_name: str = data["your_name"]
        self.your_email: str = data["your_email"]
        self.niche: str = data["niche"]
        self.platforms: list[str] = data.get("platforms", ["TikTok", "Instagram"])
        self.followers: str = data.get("followers", "growing audience")
        self.target_businesses: list[str] = data.get("target_businesses", [self.niche])
        self.daily_limit: int = data.get("daily_limit", 50)
        self.delay_seconds: int = data.get("delay_between_emails", 30)
        self.results_file: str = data.get("results_file", "results.csv")

        self.anthropic_api_key: str = data.get("anthropic_api_key", "")
        self.google_api_key: str = data.get("google_api_key", "")
        self.google_cx: str = data.get("google_cx", "")

        self.smtp_host: str = data.get("smtp_host", "smtp.gmail.com")
        self.smtp_port: int = data.get("smtp_port", 465)
        self.smtp_password: str = data.get("smtp_password", "")


class Searcher:
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UGCBot/1.0; +https://ugcagent.io)"}

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def find_businesses(self, query: str, num: int = 20) -> list[dict]:
        if self.cfg.google_api_key and self.cfg.google_cx:
            return self._google(query, num)
        return self._duckduckgo(query, num)

    def _google(self, query: str, num: int) -> list[dict]:
        results = []
        for start in range(1, num + 1, 10):
            try:
                resp = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": self.cfg.google_api_key,
                        "cx": self.cfg.google_cx,
                        "q": query,
                        "start": start,
                        "num": min(10, num - len(results)),
                    },
                    timeout=12,
                )
                resp.raise_for_status()
                for item in resp.json().get("items", []):
                    results.append({
                        "business_name": _clean_title(item.get("title", "")),
                        "website": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    })
                if len(results) >= num:
                    break
            except Exception as exc:
                log.warning(f"Google search error: {exc}")
                break
        return results

    def _duckduckgo(self, query: str, num: int) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num):
                    results.append({
                        "business_name": _clean_title(r.get("title", "")),
                        "website": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results
        except ImportError:
            log.error("Install duckduckgo-search: pip install duckduckgo-search")
            return []
        except Exception as exc:
            log.error(f"DuckDuckGo search error: {exc}")
            return []


class EmailExtractor:
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UGCBot/1.0)"}

    def extract(self, url: str) -> str | None:
        for page_url in self._pages_to_try(url):
            email = self._scan_page(page_url)
            if email:
                return email
        return None

    def _pages_to_try(self, base_url: str) -> list[str]:
        base = base_url.rstrip("/")
        pages = [base]
        for path in CONTACT_PATHS:
            pages.append(base + path)
        return pages

    def _scan_page(self, url: str) -> str | None:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10, allow_redirects=True)
            if not resp.ok:
                return None
            soup = BeautifulSoup(resp.text, "html.parser")

            # mailto: links are most reliable
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().startswith("mailto:"):
                    email = href[7:].split("?")[0].strip().lower()
                    if _valid_email(email):
                        return email

            # Fallback: regex scan
            for email in EMAIL_RE.findall(resp.text):
                if _valid_email(email.lower()):
                    return email.lower()

        except Exception as exc:
            log.debug(f"Scan failed for {url}: {exc}")
        return None


class EmailWriter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, business_name: str, website: str, snippet: str) -> tuple[str, str]:
        if self.cfg.anthropic_api_key:
            return self._claude(business_name, website, snippet)
        return self._template(business_name, website)

    def _claude(self, business_name: str, website: str, snippet: str) -> tuple[str, str]:
        import anthropic as sdk
        client = sdk.Anthropic(api_key=self.cfg.anthropic_api_key)

        prompt = (
            f"You are {self.cfg.your_name}, a UGC creator in the {self.cfg.niche} niche "
            f"on {', '.join(self.cfg.platforms)} with {self.cfg.followers} followers.\n\n"
            f"Write a short, personalized cold outreach email to this brand:\n"
            f"- Name: {business_name}\n"
            f"- Website: {website}\n"
            f"- About: {snippet}\n\n"
            f"Rules:\n"
            f"- Subject: specific to their brand, compelling, under 60 chars\n"
            f"- Body: 3-5 short sentences, reference something specific about them\n"
            f"- Offer: propose creating UGC content (videos/photos) for them\n"
            f"- Close with your reply email: {self.cfg.your_email}\n"
            f"- Tone: genuine, conversational, not salesy\n"
            f"- No emojis in subject line\n\n"
            f'Return ONLY valid JSON: {{"subject": "...", "body": "..."}}'
        )

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"Bad AI response: {text[:100]}")
        data = json.loads(match.group())
        return data["subject"], data["body"]

    def _template(self, business_name: str, website: str) -> tuple[str, str]:
        subject = f"UGC collaboration with {business_name}?"
        body = (
            f"Hi {business_name} team,\n\n"
            f"I came across {website} and love what you're doing.\n\n"
            f"I'm {self.cfg.your_name}, a UGC creator in the {self.cfg.niche} niche "
            f"on {', '.join(self.cfg.platforms)} with {self.cfg.followers} followers. "
            f"I'd love to create authentic content for your brand.\n\n"
            f"Would you be open to a quick collaboration? Reply here or reach me at {self.cfg.your_email}.\n\n"
            f"Best,\n{self.cfg.your_name}"
        )
        return subject, body


class ResultsLogger:
    def __init__(self, path: str):
        self.path = path
        if not Path(path).exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=RESULTS_HEADERS).writeheader()

    def write(self, **kwargs):
        row = {h: kwargs.get(h, "") for h in RESULTS_HEADERS}
        row["sent_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RESULTS_HEADERS).writerow(row)


class UGCAgent:
    def __init__(self, config_path: str):
        self.cfg = Config(config_path)
        self.searcher = Searcher(self.cfg)
        self.extractor = EmailExtractor()
        self.writer = EmailWriter(self.cfg)
        self.logger = ResultsLogger(self.cfg.results_file)
        self._seen_emails: set[str] = set()

    def run(self, dry_run: bool = False) -> int:
        cfg = self.cfg
        sent = 0
        log.info(
            f"UGC Agent starting | Niche: {cfg.niche} | "
            f"Limit: {cfg.daily_limit}/day | Dry-run: {dry_run}"
        )

        for target in cfg.target_businesses:
            if sent >= cfg.daily_limit:
                break

            query = f"{target} business email collaboration {cfg.niche}"
            log.info(f"Searching: {query}")
            businesses = self.searcher.find_businesses(query, num=20)
            log.info(f"  Found {len(businesses)} results")

            for biz in businesses:
                if sent >= cfg.daily_limit:
                    break
                sent += self._process(biz, dry_run)

        log.info(f"Done. Emails {'would be sent' if dry_run else 'sent'}: {sent}")
        return sent

    def _process(self, biz: dict, dry_run: bool) -> int:
        name = biz["business_name"] or "Unknown"
        url = biz.get("website", "")

        if not url or not url.startswith("http"):
            return 0

        log.info(f"Processing: {name} ({url})")

        email = self.extractor.extract(url)
        if not email:
            log.info("  No email found — skipping")
            self.logger.write(business_name=name, website=url, status="no_email")
            return 0

        if email in self._seen_emails:
            log.info(f"  Duplicate email {email} — skipping")
            return 0
        self._seen_emails.add(email)

        log.info(f"  Email: {email}")

        try:
            subject, body = self.writer.generate(name, url, biz.get("snippet", ""))
        except Exception as exc:
            log.error(f"  Email generation failed: {exc}")
            self.logger.write(business_name=name, website=url, email=email, status="gen_error")
            return 0

        if dry_run:
            log.info(f"  [DRY RUN] Subject: {subject}")
            log.info(f"  [DRY RUN] Body preview: {body[:80]}...")
            self.logger.write(
                business_name=name, website=url, email=email,
                status="dry_run", subject=subject, message_preview=body[:120],
            )
            return 1

        ok = self._send(email, subject, body)
        status = "sent" if ok else "send_failed"
        self.logger.write(
            business_name=name, website=url, email=email,
            status=status, subject=subject, message_preview=body[:120],
        )

        if ok:
            log.info(f"  Sent! Waiting {self.cfg.delay_seconds}s...")
            time.sleep(self.cfg.delay_seconds)
            return 1

        return 0

    def _send(self, to: str, subject: str, body: str) -> bool:
        cfg = self.cfg
        try:
            msg = MIMEMultipart()
            msg["From"] = cfg.your_email
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port) as server:
                server.login(cfg.your_email, cfg.smtp_password)
                server.send_message(msg)
            return True
        except Exception as exc:
            log.error(f"  SMTP error: {exc}")
            return False


def _clean_title(title: str) -> str:
    for sep in (" - ", " | ", " – ", " — ", " :: "):
        if sep in title:
            return title.split(sep)[0].strip()
    return title.strip()


def _valid_email(email: str) -> bool:
    if not EMAIL_RE.match(email):
        return False
    domain = email.split("@")[-1].lower()
    if domain in THROWAWAY_DOMAINS:
        return False
    if email.startswith("no-reply@") or email.startswith("noreply@"):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="UGC Brand Deal AI Agent — auto-finds brands and sends personalized pitches"
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find emails and generate messages but do NOT send",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(
            f"Error: '{args.config}' not found.\n"
            "Copy config.example.json → config.json and fill in your details."
        )
        sys.exit(1)

    agent = UGCAgent(args.config)
    agent.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

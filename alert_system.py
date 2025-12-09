import smtplib
import requests
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

class AlertManager:
    def __init__(self, discord_webhook_url: str = None, 
                 email_config: dict = None):
        """
        Initialize AlertManager.
        
        email_config structure:
        {
            'sender_email': '...',
            'sender_password': '...', # App Password
            'receiver_email': '...',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587
        }
        """
        self.discord_url = discord_webhook_url
        self.email_config = email_config
        
    def generate_chart_image(self, data: pd.DataFrame, ticker: str, lookback: int = 50,
                             equity_curve: pd.Series = None, features: pd.DataFrame = None,
                             probs: pd.DataFrame = None, signals: pd.Series = None,
                             meta_signals: pd.Series = None, meta_equity_curve: pd.Series = None) -> io.BytesIO:
        """
        Generate a comprehensive chart image (Dashboard style).
        Supports Comparison Mode (Raw vs Meta).
        """
        # Slice last N candles
        df = data.tail(lookback).copy()
        
        # Determine layout
        # Standard: Price, Equity, Super, Volume
        # Meta Mode: Price(Raw), Price(Meta), Equity(Comp), Super, Volume
        
        has_meta = meta_signals is not None and meta_equity_curve is not None
        has_equity = equity_curve is not None
        has_super = features is not None and probs is not None
        
        # Define Ratios
        # Raw Mode: Price(3), Eq(1), Sup(1), Vol(1) -> 6
        # Meta Mode: PriceRaw(2), PriceMeta(2), Eq(1), Sup(1), Vol(1) -> 7
        
        plot_ratios = [3]
        if has_meta:
            plot_ratios = [2, 2] # Two price charts
            
        if has_equity: plot_ratios.append(1)
        if has_super: plot_ratios.append(1)
        plot_ratios.append(1) # Volume
        
        total_height = 8 + (len(plot_ratios)-2)*2
        if has_meta: total_height += 2 # Taller for extra chart
        
        fig, axes = plt.subplots(len(plot_ratios), 1, figsize=(12, total_height), 
                                 gridspec_kw={'height_ratios': plot_ratios}, sharex=True)
        
        if len(plot_ratios) == 1: axes = [axes]
        
        ax_idx = 0
        
        # --- 1. Raw Price Chart ---
        ax_price = axes[ax_idx]
        ax_idx += 1
        
        ax_price.plot(df.index, df['close'], label='Close', color='white', linewidth=1.0)
        ax_price.fill_between(df.index, df['low'], df['high'], color='gray', alpha=0.3)
        
        if 'sma_20' in df.columns:
            ax_price.plot(df.index, df['sma_20'], label='SMA 20', color='cyan', linestyle='--', alpha=0.5)
            
        # Plot Signals (Raw)
        sig_slice = signals.reindex(df.index).fillna(0) if signals is not None else pd.Series(0, index=df.index)
        
        buys = sig_slice[sig_slice == 1]
        sells = sig_slice[sig_slice == -1]
        
        if not buys.empty:
            ax_price.scatter(buys.index, df.loc[buys.index, 'low']*0.99, marker='^', color='#00ff00', s=80, label='Buy (Raw)', zorder=5)
        if not sells.empty:
            ax_price.scatter(sells.index, df.loc[sells.index, 'high']*1.01, marker='v', color='#ff0000', s=80, label='Sell (Raw)', zorder=5)

        ax_price.set_title(f"{ticker} - Raw Model Signals", color='white', fontsize=12, fontweight='bold')
        ax_price.grid(True, color='#333333')
        ax_price.set_facecolor('#0e0e0e')
        ax_price.legend(loc='upper left', framealpha=0.2, fontsize='small')
        
        # --- 2. Meta Price Chart (If Enabled) ---
        if has_meta:
            ax_meta = axes[ax_idx]
            ax_idx += 1
            
            ax_meta.plot(df.index, df['close'], color='white', linewidth=1.0, alpha=0.7)
            ax_meta.fill_between(df.index, df['low'], df['high'], color='gray', alpha=0.2)
            
            # Plot Meta Signals
            meta_slice = meta_signals.reindex(df.index).fillna(0)
            
            # Accepted
            m_buys = meta_slice[meta_slice == 1]
            m_sells = meta_slice[meta_slice == -1]
            
            # Rejected (Raw was 1/-1 but Meta is 0)
            # Need to align raw slice
            rej_buys = (sig_slice == 1) & (meta_slice == 0)
            rej_sells = (sig_slice == -1) & (meta_slice == 0)
            
            # Plot Accepted
            if not m_buys.empty:
                ax_meta.scatter(m_buys.index, df.loc[m_buys.index, 'low']*0.99, marker='^', color='#00ff00', s=80, label='Accepted Buy', zorder=5)
            if not m_sells.empty:
                ax_meta.scatter(m_sells.index, df.loc[m_sells.index, 'high']*1.01, marker='v', color='#ff0000', s=80, label='Accepted Sell', zorder=5)
                
            # Plot Rejected
            r_buys_idx = rej_buys[rej_buys].index
            r_sells_idx = rej_sells[rej_sells].index
            
            if not r_buys_idx.empty:
                ax_meta.scatter(r_buys_idx, df.loc[r_buys_idx, 'low']*0.99, marker='x', color='#ff4444', s=60, label='Rejected Buy', zorder=4)
            if not r_sells_idx.empty:
                ax_meta.scatter(r_sells_idx, df.loc[r_sells_idx, 'high']*1.01, marker='x', color='#ffaa00', s=60, label='Rejected Sell', zorder=4)
                
            ax_meta.set_title(f"Meta-Model Decisions (Accepted vs Rejected)", color='white', fontsize=12, fontweight='bold')
            ax_meta.grid(True, color='#333333')
            ax_meta.set_facecolor('#0e0e0e')
            ax_meta.legend(loc='upper left', framealpha=0.2, fontsize='small')

        # --- 3. Equity Curve ---
        if has_equity:
            ax_eq = axes[ax_idx]
            ax_idx += 1
            
            eq_slice = equity_curve.reindex(df.index).ffill()
            
            # Plot Raw
            if not eq_slice.empty:
                start_val = eq_slice.iloc[0]
                norm_eq = (eq_slice - start_val) / start_val * 100
                ax_eq.plot(norm_eq.index, norm_eq, color='gray', linestyle='--', label='Raw Equity %')
                
            # Plot Meta
            if has_meta:
                meta_eq_slice = meta_equity_curve.reindex(df.index).ffill()
                if not meta_eq_slice.empty:
                    m_start = meta_eq_slice.iloc[0]
                    m_norm = (meta_eq_slice - m_start) / m_start * 100
                    color = '#00ff00' if m_norm.iloc[-1] > 0 else '#ff4444'
                    ax_eq.plot(m_norm.index, m_norm, color=color, linewidth=2.0, label='Meta Equity %')
            
            ax_eq.set_ylabel("Return %", color='white')
            ax_eq.grid(True, color='#333333')
            ax_eq.set_facecolor('#0e0e0e')
            ax_eq.legend(loc='upper left', framealpha=0.2, fontsize='small')

        # --- 4. Super Indicator ---
        if has_super:
            ax_sup = axes[ax_idx]
            ax_idx += 1
            # ... (Existing Logic) ...
            f_slice = features.reindex(df.index).fillna(0)
            p_slice = probs.reindex(df.index).fillna(0)
            ml_sent = p_slice['prob_buy'] - p_slice['prob_sell']
            comp = f_slice['composite_oscillator'] if 'composite_oscillator' in f_slice.columns else pd.Series(0, index=df.index)
            
            ax_sup.plot(comp.index, comp, label='Composite', color='orange', linestyle=':', linewidth=1.0)
            ax_sup.plot(ml_sent.index, ml_sent, label='Confidence', color='#44aaff', linewidth=1.5)
            
            ax_sup.axhline(0, color='gray', alpha=0.5)
            ax_sup.fill_between(ml_sent.index, ml_sent, 0, where=(ml_sent>=0), color='#44aaff', alpha=0.2)
            ax_sup.fill_between(ml_sent.index, ml_sent, 0, where=(ml_sent<0), color='#ff4444', alpha=0.2)
            
            ax_sup.set_ylabel("Ind", color='white')
            ax_sup.grid(True, color='#333333')
            ax_sup.set_facecolor('#0e0e0e')
            ax_sup.legend(loc='upper left', framealpha=0.2, fontsize='small')

        # --- 5. Volume ---
        ax_vol = axes[ax_idx]
        colors = ['#00ff00' if c >= o else '#ff0000' for c, o in zip(df['close'], df['open'])]
        ax_vol.bar(df.index, df['volume'], color=colors, alpha=0.6)
        ax_vol.set_ylabel("Vol", color='white')
        ax_vol.grid(True, color='#333333')
        ax_vol.set_facecolor('#0e0e0e')
        
        # Formatting
        fig.patch.set_facecolor('black')
        for ax in axes:
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return buf

    def send_alert(self, title: str, message: str, data: pd.DataFrame = None, ticker: str = "Unknown", 
                   color: int = 0x00ff00, fields: list = None, chart_lookback: int = 180,
                   equity_curve: pd.Series = None, features: pd.DataFrame = None, probs: pd.DataFrame = None,
                   signals: pd.Series = None, meta_signals: pd.Series = None, meta_equity_curve: pd.Series = None):
        """
        Send alert with optional Comparison Charts.
        """
        chart_buf = None
        if data is not None:
            try:
                chart_buf = self.generate_chart_image(
                    data, ticker, lookback=chart_lookback,
                    equity_curve=equity_curve, features=features, probs=probs,
                    signals=signals, meta_signals=meta_signals, meta_equity_curve=meta_equity_curve
                )
            except Exception as e:
                print(f"Failed to generate chart: {e}")
                import traceback
                traceback.print_exc()
        
        # Send Discord
        if self.discord_url:
            self._send_discord(title, message, chart_buf, color, fields)
            
        # Send Email
        if self.email_config:
            self._send_email(title, message, chart_buf, fields)
            
    def _send_discord(self, title, message, chart_buf, color, fields):
        try:
            # Discord Webhook with Embed
            payload = {
                "username": "PatternFindR Bot",
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": "PatternFindR Live Trading System"}
                }]
            }
            
            if fields:
                payload["embeds"][0]["fields"] = fields
                
            files = {}
            if chart_buf:
                chart_buf.seek(0)
                files = {'file': ('chart.png', chart_buf, 'image/png')}
                # Add image to embed
                payload["embeds"][0]["image"] = {"url": "attachment://chart.png"}
            
            # If using files, payload must be sent as multipart/form-data, 
            # but 'embeds' field in multipart is complex.
            # Easier way: Send payload as JSON_payload field in multipart
            
            if files:
                response = requests.post(
                    self.discord_url, 
                    files=files,
                    data={'payload_json': json.dumps(payload)}
                )
            else:
                response = requests.post(
                    self.discord_url, 
                    json=payload
                )
                
            if response.status_code not in [200, 204]:
                print(f"Discord Send Failed: {response.status_code} {response.text}")
                
        except Exception as e:
            print(f"Error sending Discord alert: {e}")

    def _send_email(self, title, message, chart_buf, fields):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['receiver_email']
            msg['Subject'] = title
            
            # HTML Body
            html_body = f"<h2>{title}</h2><p>{message}</p>"
            
            if fields:
                html_body += "<h3>Stats:</h3><ul>"
                for f in fields:
                    html_body += f"<li><b>{f['name']}:</b> {f['value']}</li>"
                html_body += "</ul>"
                
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach Image
            if chart_buf:
                chart_buf.seek(0)
                img = MIMEImage(chart_buf.read())
                img.add_header('Content-Disposition', 'attachment', filename="chart.png")
                msg.attach(img)
            
            # Send
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f"Error sending Email alert: {e}")

# Test block
if __name__ == "__main__":
    # Example usage
    print("AlertManager loaded.")

/**
 * Resend API client for sending emails.
 */

import { RESEND_API_KEY, RESEND_FROM } from "../config.js";

const RESEND_API_BASE = "https://api.resend.com";

/**
 * Send an email via Resend API.
 */
export const sendEmail = async ({ to, subject, html, text, replyTo }) => {
  const recipients = Array.isArray(to) ? to : [to];

  const body = {
    from: RESEND_FROM,
    to: recipients,
    subject,
    ...(html && { html }),
    ...(text && { text }),
    ...(replyTo && { reply_to: replyTo }),
  };

  const response = await fetch(`${RESEND_API_BASE}/emails`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${RESEND_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.message || `Resend API error: ${response.status}`);
  }

  return data;
};

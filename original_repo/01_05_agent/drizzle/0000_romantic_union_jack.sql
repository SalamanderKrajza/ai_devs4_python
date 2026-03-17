CREATE TABLE `agents` (
	`id` text PRIMARY KEY NOT NULL,
	`session_id` text NOT NULL,
	`root_agent_id` text NOT NULL,
	`parent_id` text,
	`source_call_id` text,
	`depth` integer DEFAULT 0 NOT NULL,
	`task` text NOT NULL,
	`config` text NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`waiting_for` text DEFAULT '[]' NOT NULL,
	`result` text,
	`error` text,
	`turn_count` integer DEFAULT 0 NOT NULL,
	`created_at` integer NOT NULL,
	`started_at` integer,
	`completed_at` integer,
	FOREIGN KEY (`session_id`) REFERENCES `sessions`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `agents_session_idx` ON `agents` (`session_id`);--> statement-breakpoint
CREATE INDEX `agents_parent_idx` ON `agents` (`parent_id`);--> statement-breakpoint
CREATE INDEX `agents_status_idx` ON `agents` (`status`);--> statement-breakpoint
CREATE TABLE `items` (
	`id` text PRIMARY KEY NOT NULL,
	`agent_id` text NOT NULL,
	`sequence` integer NOT NULL,
	`type` text NOT NULL,
	`role` text,
	`content` text,
	`call_id` text,
	`name` text,
	`arguments` text,
	`output` text,
	`is_error` integer,
	`summary` text,
	`signature` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`agent_id`) REFERENCES `agents`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `items_agent_idx` ON `items` (`agent_id`);--> statement-breakpoint
CREATE INDEX `items_agent_seq_idx` ON `items` (`agent_id`,`sequence`);--> statement-breakpoint
CREATE INDEX `items_call_id_idx` ON `items` (`call_id`);--> statement-breakpoint
CREATE TABLE `sessions` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`root_agent_id` text,
	`title` text,
	`status` text DEFAULT 'active' NOT NULL,
	`created_at` integer NOT NULL,
	`updated_at` integer,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `sessions_user_idx` ON `sessions` (`user_id`);--> statement-breakpoint
CREATE TABLE `users` (
	`id` text PRIMARY KEY NOT NULL,
	`email` text NOT NULL,
	`api_key_hash` text NOT NULL,
	`created_at` integer NOT NULL,
	`updated_at` integer
);
--> statement-breakpoint
CREATE UNIQUE INDEX `users_email_idx` ON `users` (`email`);--> statement-breakpoint
CREATE UNIQUE INDEX `users_api_key_hash_idx` ON `users` (`api_key_hash`);
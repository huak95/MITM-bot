import {
  SlashCommandBuilder,
  EmbedBuilder,
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  ComponentType,
} from "discord.js";
import { getVoiceConnection } from "@discordjs/voice";
import { createWriteStream } from "fs";
import prism from "prism-media";
import ffmpeg from "fluent-ffmpeg";
import path from "path";

// Reference to the active sessions (should be the same object used in join command)
import { activeSessions } from "./join.js";

// Object to store active recordings
const activeRecordings = new Map();

export const data = new SlashCommandBuilder()
  .setName("record")
  .setDescription("Starts an interactive recording session");

export async function execute(interaction) {
  const sessionInfo = activeSessions.get(interaction.guild.id);
  if (!sessionInfo) {
    return interaction.reply("No active voice session found. Use /join first!");
  }

  let isRecording = false;
  let recordStartTime;

  const embed = new EmbedBuilder()
    .setColor("#0099ff")
    .setTitle("Recording Session")
    .addFields(
      { name: "Session ID", value: sessionInfo.id },
      { name: "Channel", value: sessionInfo.channel },
      { name: "Status", value: "Ready to record" },
      {
        name: "Activity",
        value: sessionInfo.activity.slice(-5).join("\n") || "No activity",
      }
    );

  const startButton = new ButtonBuilder()
    .setCustomId("start_recording")
    .setLabel("Start Recording")
    .setStyle(ButtonStyle.Success);

  const stopButton = new ButtonBuilder()
    .setCustomId("stop_recording")
    .setLabel("Stop Recording")
    .setStyle(ButtonStyle.Danger)
    .setDisabled(true);

  const row = new ActionRowBuilder().addComponents(startButton, stopButton);

  const response = await interaction.reply({
    embeds: [embed],
    components: [row],
    fetchReply: true,
  });

  const collector = response.createMessageComponentCollector({
    componentType: ComponentType.Button,
    time: 3600000, // 1 hour
  });

  collector.on("collect", async (i) => {
    if (i.customId === "start_recording") {
      if (!isRecording) {
        isRecording = true;
        recordStartTime = new Date();
        sessionInfo.activity.push(
          `${getTimeDiff(
            sessionInfo.startTime,
            recordStartTime
          )}: Recording started by @${i.user.username}.`
        );

        startButton.setDisabled(true);
        stopButton.setDisabled(false);
        updateEmbed();

        // Start the actual recording process
        startRecording(interaction.guild.id, sessionInfo.id);
      }
    } else if (i.customId === "stop_recording") {
      if (isRecording) {
        isRecording = false;
        const endTime = new Date();
        sessionInfo.activity.push(
          `${getTimeDiff(
            sessionInfo.startTime,
            endTime
          )}: Recording stopped by @${i.user.username}.`
        );

        startButton.setDisabled(false);
        stopButton.setDisabled(true);
        updateEmbed(true);

        // Stop the actual recording process
        const filename = stopRecording(interaction.guild.id);
        if (filename) {
          await i.followUp(`Recording saved as: ${filename}`);
        }
      }
    }
    await i.update({ embeds: [embed], components: [row] });
  });

  function updateEmbed(isFinal = false) {
    // ... (previous updateEmbed function remains the same)
  }

  function getTimeDiff(start, end) {
    const diff = (end - start) / 1000;
    return new Date(diff * 1000).toISOString().substr(11, 8);
  }
}

function startRecording(guildId, sessionId) {
  const connection = getVoiceConnection(guildId);
  if (!connection) return;

  const receiver = connection.receiver;
  const audioStream = receiver.subscribe(undefined, {
    end: {
      behavior: "manual",
    },
  });

  const filename = path.join(
    __dirname,
    "..",
    "recordings",
    `${sessionId}-${Date.now()}.mp3`
  );
  console.log(`Started recording: ${filename}`);

  const opusDecoder = new prism.opus.Decoder({
    rate: 48000,
    channels: 2,
    frameSize: 960,
  });

  const ffmpegProcess = ffmpeg(audioStream.pipe(opusDecoder))
    .inputFormat("s32le")
    .audioFrequency(48000)
    .audioChannels(2)
    .audioCodec("libmp3lame")
    .audioBitrate("128k")
    .outputOptions("-af", "aresample=48000")
    .format("mp3")
    .on("error", (err) => {
      console.error(`An error occurred: ${err.message}`);
    })
    .on("end", () => {
      console.log("Processing finished!");
    });

  ffmpegProcess.save(filename);

  activeRecordings.set(guildId, { ffmpegProcess, filename });
}

function stopRecording(guildId) {
  const recording = activeRecordings.get(guildId);
  if (recording) {
    recording.ffmpegProcess.kill("SIGINT");
    activeRecordings.delete(guildId);
    return recording.filename;
  }
  return null;
}

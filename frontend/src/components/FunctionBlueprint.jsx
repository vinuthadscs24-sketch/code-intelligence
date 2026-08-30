import React, { useState } from 'react';
import { Code2, ArrowDownRight, Layers, CheckCircle2, Copy } from 'lucide-react';

export default function FunctionBlueprint({ data, rawCode }) {
  const [copied, setCopied] = useState(false);

  const functionName = data?.function_name || "bookAppointment";
  const purpose = data?.purpose || "Creates and persists a new appointment after validating patient and doctor availability.";
  const inputs = data?.inputs || ["patientId: String", "doctorId: String", "date: LocalDateTime"];
  const flowSteps = data?.flow_steps || [
    "Validate input parameters",
    "Check doctor schedule availability",
    "Construct Appointment entity",
    "Persist via AppointmentRepository",
    "Emit AppointmentCreatedEvent"
  ];
  const files = data?.files || ["AppointmentService.java", "AppointmentRepository.java"];

  const handleCopy = () => {
    navigator.clipboard.writeText(rawCode || "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full h-full bg-[#090D16] p-6 overflow-y-auto flex flex-col gap-6">
      
      {/* Top Header & Metadata */}
      <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded text-xs font-mono font-semibold uppercase">
              Function Blueprint
            </span>
            <h2 className="text-xl font-bold font-mono text-white">{functionName}()</h2>
          </div>
          <div className="flex items-center gap-2">
            {files.map((file, idx) => (
              <span key={idx} className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded border border-slate-700">
                📄 {file}
              </span>
            ))}
          </div>
        </div>
        <p className="text-slate-300 text-sm leading-relaxed">{purpose}</p>
      </div>

      {/* Grid: Inputs & Structured Execution Flow */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Left Col: Inputs / Signature */}
        <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <Layers className="w-4 h-4 text-blue-400" /> Inputs / Signature
          </div>
          <div className="flex flex-col gap-2 mt-1">
            {inputs.map((arg, idx) => (
              <div key={idx} className="bg-[#0B0F17] border border-slate-800 p-2.5 rounded text-xs font-mono text-blue-300 flex items-center justify-between">
                <span>{arg}</span>
                <span className="text-slate-600 text-[10px]">Param {idx + 1}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right Col: Internal Flow */}
        <div className="md:col-span-2 bg-[#131B2E] border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <ArrowDownRight className="w-4 h-4 text-blue-400" /> Internal Execution Flow
          </div>
          <div className="flex flex-col gap-2 mt-1">
            {flowSteps.map((step, idx) => (
              <div key={idx} className="flex items-center gap-3 bg-[#0B0F17] border border-slate-800/80 p-2.5 rounded-lg">
                <span className="w-6 h-6 rounded-full bg-blue-600/20 text-blue-400 font-mono text-xs font-bold flex items-center justify-center border border-blue-500/30">
                  {idx + 1}
                </span>
                <span className="text-slate-200 text-xs font-medium">{step}</span>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* Embedded Relevant Code */}
      <div className="bg-[#131B2E] border border-slate-800 rounded-xl overflow-hidden">
        <div className="bg-[#1A243B] px-5 py-3 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
            <Code2 className="w-4 h-4 text-blue-400" /> Implementation Source Code
          </div>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition"
          >
            {copied ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? "Copied" : "Copy Code"}</span>
          </button>
        </div>
        <pre className="p-5 font-mono text-xs text-slate-300 bg-[#0B0F17] overflow-x-auto leading-relaxed">
          <code>{rawCode || `public Appointment bookAppointment(String patientId, String doctorId, LocalDateTime date) {\n    validate(patientId, doctorId, date);\n    boolean available = scheduleRepository.isDoctorAvailable(doctorId, date);\n    if (!available) throw new ScheduleConflictException();\n    \n    Appointment appt = new Appointment(patientId, doctorId, date);\n    return appointmentRepository.save(appt);\n}`}</code>
        </pre>
      </div>

    </div>
  );
}